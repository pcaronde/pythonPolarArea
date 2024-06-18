import abc
import difflib
import enum
import hashlib
import json
import logging
import nh3
import re
import string
import time
import traceback
import typing
import uuid
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import wraps, reduce
from itertools import chain
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Collection,
    Union,
    Set,
    Iterable,
)

from codility_task_spec.id import Id
from codility_task_spec.variant import Variant
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.contrib.sessions.backends.base import SessionBase
from django.core.exceptions import (
    MultipleObjectsReturned,
    ObjectDoesNotExist,
    ValidationError,
)
from django.db import connection, IntegrityError, models, transaction
from django.db.models import Q, signals, QuerySet
from django.db.transaction import atomic
from django.template import Context, Template
from django.urls import reverse
from django.utils.functional import cached_property
from model_utils import Choices
from opentelemetry.trace import get_tracer
from pygments.util import ClassNotFound
from requests import RequestException

import jsonfield
from codility.backend import checker_api
from codility.candidate.api.parsers import parse_options_string
from codility.candidate.evaluation_solutions_client import (
    EvaluationsSolutionS3Client,
    PresignedUrlExpiredError,
    SolutionFiles,
)
from codility.candidate.evaluations_api_client import (
    EvaluationsApiClientHttp,
    EvaluationApiClientException,
)
from codility.codelive.user_sessions import is_codelive_interviewer
from codility.languages.models import PRG_LANGS, prg_lang_visible_in_programmers_home
from codility.payments.models import Package
from codility.permissions import FeatureMapper
from codility.permissions.role import UserRole
from codility.privacy.models import Identity
from codility.profiles.models import (
    AccountGroup,
    Team,
    User,
    CustomAnonymousUser,
    exclude_report_only_recipients,
)
from codility.profiles.public import get_role_for_user
from codility.services.assii.assii_client import AssiiClientBuilder
from codility.services.assii.logical_models import (
    AssessmentEventType,
)
from codility.signals import event_signal, similarity_update, send_signal
from codility.structured_logger import StructuredLogger
from codility.tasks.compat import get_programming_language_variant_name
from codility.tasks.models import render_mcq_questions, Task
from codility.tasks.public import (
    change_context_from_frontend_to_task_server_format,
    get_task_info_from_task_name,
    TaskInfo,
    clean_task_context_for_task_info,
    TasksNotFoundError,
    get_task_highlight,
)
from codility.tasks.task_context import clean_task_context
from codility.tests.api.task_weights import TaskWeightDataclass
from codility.tickets.annotations import (
    TicketPossibleActionsAnnotations,
    TicketStatusAnnotations,
)
from codility.tickets.emails import render_email_report, RenderedEmailContent
from codility.tickets.options import TicketOptionField
from codility.tickets.signals import TicketSignalDispatcher
from codility.tickets.utils.task_context import TaskContextGetter
from codility.tickets.utils.variants import TicketTaskVariantHandler
from codility.utils.common import (
    bool_to_str,
    custom_sql,
    dict_remove_nones,
    dict_to_str,
    dt_to_timestamp,
    get_site_url,
    int_to_str,
    list_to_str,
    random_text,
    str_to_bool,
    str_to_dict,
    str_to_int,
    str_to_list,
    timestamp_to_dt,
)
from codility.utils.email import send_system_email
from codility.utils.log import log_entry
from codility.utils.math import convert_float_to_int_rounding_up
from codility.utils.memo import memo, reset_memo
from codility.utils.models import QuerySetAnnotation
from codility_shared.api.checker import CheckerApi
from codility_shared.codelive_task_config import (
    CODELIVE_WHITEBOARD_TEMPLATE_TASK,
)
from codility_shared.report.parse import get_summary, get_test_groups, parse_xml_rpt

if typing.TYPE_CHECKING:
    from codility.campaigns.models import TaskWeight
    from codility.tickets.utils.ticket_builder import TaskProvider, PackageProvider

TICKET_DEFAULT_TIMELIMIT = 30 * 60  # 30 minutes
SHARE_TOKEN_DEFAULT_DURATION = 7  # 7 days

SUBMIT_REPORT_FROM_EVALUATIONS_API_ENABLED_AG = (
    "Submit report from Evaluations API enabled"
)

logger = typing.cast(StructuredLogger, logging.getLogger(__name__))

trace = get_tracer(__name__)


def pairwise(seq):
    return zip(seq, seq[1:])  # [1,2,3,4] -> (1,2), (2,3), (3,4)


def to_minutes(seconds):
    if seconds is not None:
        return (seconds + 59) // 60
    else:
        return None


class TicketTaskInfo(dict):
    name: str
    prg_lang_list: Optional[List[str]]
    task_weight_id: Optional[str]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        keys = list(self.keys())
        for key in keys:
            if self[key] is None:
                del self[key]


class TicketQuerySet(models.QuerySet):
    @trace.start_as_current_span("TicketQuerySet.visible")
    def visible(self, include_ghost=True):
        """Filter out old removed tickets and template tickets, which
        should not be externally shown anywhere."""
        res = (
            self.exclude(origin="template")
            .filter(removed=False)
            .filter(codelive_session__deleted_at__isnull=True)
        )
        if include_ghost:
            return res
        else:
            # Exclude ghost tickets
            return res.exclude(origin="public", start_date__isnull=True)

    @trace.start_as_current_span("TicketQuerySet.visible_to")
    def visible_to(
        self,
        user,
        feature_mapper: FeatureMapper,
        user_role: UserRole = None,
        team: Optional[Team] = None,
        teams_ids: Optional[List[int]] = None,
        disregard_can_see_others=False,
        all_if_staff=True,
        all_if_admin=False,
    ) -> "TicketQuerySet":
        if (user.is_superuser or user.is_staff) and all_if_staff:
            return self.visible()

        elif user.is_recruiter:
            if user_role is None:
                user_role = get_role_for_user(user)

            if team:
                teams_ids = [team.id]
            elif user.is_admin and all_if_admin:
                teams_ids = user.account.teams.values_list("id", flat=True)
            elif teams_ids is None:
                teams_ids = user.teams.values_list("id", flat=True)

            if not settings.ENABLE_TICKETS_VISIBLE_TO_NEW_FILTER:
                res = self.filter(
                    id__in=self.__get_ticket_ids(
                        user=user,
                        user_role=user_role,
                        teams=teams_ids,
                    )
                )

                if user_role.codelive_can_see_sessions_interacted_with:
                    return res

                # Filter by user permission if needed
                if not (user.can_see_others() or disregard_can_see_others):
                    res = res.filter(creator=user)
                return res

            base_qs = self.visible(include_ghost=False).only("id")
            querysets = []
            filters = Q()

            if not user_role.ticket_can_see_team_sessions or not (
                user.can_see_others() or disregard_can_see_others
            ):
                filters = (
                    Q(creator=user)
                    | Q(campaign__url_creator=user)
                    | Q(campaign__template__creator=user)
                )
            querysets.extend(
                [
                    base_qs.filter(filters, team__in=teams_ids),
                    base_qs.filter(filters, campaign__team__in=teams_ids),
                ]
            )

            if user_role.codelive_can_see_sessions_interacted_with:
                filters = Q(is_codelive=True) & (
                    Q(creator=user)
                    | Q(codelive_session__assigned_interviewers=user)
                    | Q(codelive_session__participants__user=user)
                )
                querysets.append(base_qs.filter(filters))

            result = reduce(
                TicketQuerySet.union,
                querysets,
                self.none(),
            )
            return typing.cast(TicketQuerySet, self.filter(id__in=result))

        else:
            return self.none()

    def cert_tickets(self):
        return self.visible().filter(origin="cert")

    def with_status(self):
        return TicketStatusAnnotations().annotate(self)

    def with_possible_actions(self, possible_action=None):
        return TicketPossibleActionsAnnotations().annotate(self, possible_action)

    def using_tasks(self, task_names: Collection[str]) -> "TicketQuerySet":
        # Note that this takes advantage of the fact that all tasks
        # per ticket are distinct - otherwise we would get duplicate
        # rows.
        return self.filter(tickettask__task__name__in=task_names)

    def __get_ticket_ids(
        self,
        user: User,
        user_role: UserRole,
        teams: Union[List[int], QuerySet[Team]],
    ) -> "TicketQuerySet":
        ids_from_campaigns = self.__get_ids_from_campaigns(team_ids=teams)
        ids_from_tickets = self.__get_ids_from_tickets(team_ids=teams)

        ids_from_campaigns_and_tickets = ids_from_campaigns.union(ids_from_tickets)

        if user_role.codelive_can_see_sessions_interacted_with:
            ids_from_sessions_interacted_with = (
                self.visible(include_ghost=False)
                .filter(is_codelive=True)
                .filter(
                    Q(creator=user)
                    | Q(codelive_session__assigned_interviewers=user)
                    | Q(codelive_session__participants__user=user)
                )
                .values("id")
            )

            return ids_from_sessions_interacted_with.union(
                ids_from_campaigns_and_tickets
            )

        return ids_from_campaigns_and_tickets

    def __get_ids_from_tickets(
        self,
        team_ids: List[int],
    ) -> "TicketQuerySet":
        return (
            self.visible(include_ghost=False).filter(team_id__in=team_ids).values("id")
        )

    def __get_ids_from_campaigns(
        self,
        team_ids: List[int],
    ) -> "TicketQuerySet":
        return (
            self.visible(include_ghost=False)
            .filter(campaign__team_id__in=team_ids)
            .values("id")
        )


class TicketManager(models.Manager):
    def _get(self, **kwargs):
        try:
            return self.get(**kwargs)
        except ObjectDoesNotExist:
            return None

    def get_queryset(self):
        return TicketQuerySet(self.model, using=self._db)

    def visible(self, *args, **kwargs):
        return self.get_queryset().visible(*args, **kwargs)

    def visible_to(self, *args, **kwargs):
        return self.get_queryset().visible_to(*args, **kwargs)

    def cert_tickets(self):
        return self.get_queryset().cert_tickets()

    def create_tickets(
        self,
        data: Sequence[dict],
        *,
        creator,
        task_provider: Optional["TaskProvider"] = None,
        package_provider: Optional["PackageProvider"] = None,
    ) -> List["Ticket"]:
        from codility.tickets.utils.ticket_builder import TicketBuilder

        return TicketBuilder(
            data,
            creator=creator,
            task_provider=task_provider,
            package_provider=package_provider,
        ).create()

    def create_with_random_id(self, extra_prefix="", **kwargs):
        prefix = extra_prefix + random_text(settings.TICKETS_RANDOM_ID_LENGTH)

        return self.create_with_prefix(prefix, **kwargs)

    def create(self, id, origin=None, **kwargs):
        """
        Secondary flow for creating a single ticket.
        TODO: rewrite to use TicketBuilder so that there's only 1 flow.
        """
        if origin is None:
            warnings.warn(
                "Ticket constructor called without a value for origin, 'private' assumed.",
                stacklevel=2,
            )
            origin = "private"

        from codility.tickets.utils.ticket_builder import enforce_team_invariants

        enforce_team_invariants(
            origin,
            kwargs.get("campaign"),
            kwargs.get("campaign_id"),
            kwargs.get("creator"),
            kwargs.get("team"),
        )

        task_infos = kwargs.pop("task_infos", None)
        task_names = kwargs.pop("task_names", None)

        if task_infos is not None and task_names is not None:
            raise Exception(
                "Cannot give both task_infos and task_names %s %s"
                % (task_infos, task_names)
            )
        if task_names is not None:
            task_infos = [{"name": task_name} for task_name in task_names]

        option_fields = {}
        model_fields = {}
        for key, value in kwargs.items():
            if isinstance(getattr(Ticket, key, None), TicketOptionField):
                option_fields[key] = value
            else:
                model_fields[key] = value

        identity_data = {}
        for field in Identity.FIELDS:
            if field in model_fields:
                identity_data[field] = model_fields.pop(field)

        ticket = super(TicketManager, self).create(id=id, origin=origin, **model_fields)
        if settings.ASSII_ENABLED:
            if not ticket.is_template:
                AssiiClientBuilder.build_assii_client().post_assessment_event(
                    AssessmentEventType.ASSESSMENT_CREATED, ticket
                )

        if identity_data:
            ticket.update_identity(**identity_data)

        if (
            ticket.creator
            and ticket.creator.is_recruiter
            and ticket.creator.account.has_beta_testing_enabled()
        ):
            option_fields["enable_beta_testing"] = True

        ticket.create_options(option_fields)

        if task_infos is not None:
            ticket.replace_tasks(task_infos)
        if ticket.is_billable and ticket.invoice is None:
            obj = Package.objects.get_package_for_creating_tickets(
                ticket.creator.account_id
            )
            if obj is not None:
                ticket.invoice = obj.package
                ticket.save()

        if ticket.invoice:
            # Creating the ticket might change active plan
            if ticket.creator:
                ticket.creator.account.reset()
            else:
                logger.error("ticket has invoice but no user, ticket=%s", ticket)
        return ticket

    def create_with_prefix(self, prefix, **kwargs):
        # A collision here is very unlikely, but possible.  Just in
        # case: retry a couple of times.
        r_len = settings.TICKETS_CONTROL_SUFFIX_LENGTH
        if "id_length" in kwargs:
            r_len = kwargs.pop("id_length")
        for i in range(0, settings.TICKETS_SUFFIX_RETRIES):
            id = prefix + "-" + random_text(r_len)
            ticket = self.create(id=id, **kwargs)
            if ticket is not None:
                return ticket

        # Getting here would mean settings.TICKETS_SUFFIX_RETRIES
        # consecutive collisions.  Might as well give up.
        raise TicketManager.CouldNotCreate()

    # See http://www.postgresql.org/docs/8.3/static/sql-select.html#SQL-FOR-UPDATE-SHARE

    def lock_for_update(self, id):
        """Lock a given ticket with a writer (i.e. exclusive) lock
        until the end of current transaction."""
        connection.cursor().execute(
            "SELECT * FROM tickets WHERE id=%s FOR UPDATE", [id]
        )

    def lock_for_share(self, id):
        """Lock a given ticket with a reader (i.e. non-exclusive) lock
        until the end of current transaction."""
        connection.cursor().execute("SELECT * FROM tickets WHERE id=%s FOR SHARE", [id])

    def recompute_ti_rpts_ready(self, id):
        from codility.similarity.models import TicketInspectionRpt
        from codility.similarity.tasks import ti_send_simchk_warning

        ticket = Ticket.objects.get(id=id)
        rpts_qs = TicketInspectionRpt.objects.filter(ticket=ticket, is_cancelled=False)

        if any(
            report.rpt_type == TicketInspectionRpt.RPT_SIM_SKIPPED for report in rpts_qs
        ):
            Ticket.objects.filter(id=id).update(ti_rpts_ready=True)
            send_signal(similarity_update, Ticket, ticket_id=id, status="skipped")

            logger.info_structured(
                "recompute_ti_rpts_ready - found sim_skipped status",
                ticket_id=id,
            )
            return

        all_rpts_count = rpts_qs.count()
        verified_rpts_count = rpts_qs.filter(
            manual_inspection_date__isnull=False
        ).count()
        # in next version we will also inspect number of rpts accepted by the customer
        if all_rpts_count > 0:
            if all_rpts_count == verified_rpts_count:
                new_value = True
            else:
                new_value = False
        else:
            new_value = None
        logger.info_structured(
            f"recompute_ti_rpts_ready - new_value={new_value}",
            ticket_id=id,
            all_reports_count=all_rpts_count,
            verified_reports_count=verified_rpts_count,
        )

        Ticket.objects.filter(id=id).update(ti_rpts_ready=new_value)

        if not new_value:
            # Hack: if there is a sim_skipped report, we should send the signal
            # this means that similarity service has found the ticket, but it deemed
            # too different to be considered as a match. It is a peculiar set of parameters,
            # but at the moment we can do nothing about it.
            if TicketInspectionRpt.objects.filter(
                ticket=ticket,
                is_cancelled=True,
                manual_inspection_result="false_not_sure",
                rpt_type="sim_skipped",
            ).exists():
                send_signal(similarity_update, Ticket, ticket_id=id, status="not-found")

            return

        if not ticket.creator:
            return

        ticket_creator: User = ticket.creator

        if (
            not ticket_creator.is_recruiter
            or not ticket_creator.account.can_use_similarity_check()
        ):
            return

        logger.info_structured(
            "recompute_ti_rpts_ready - sending similarity result for ticket",
            ticket_id=id,
            similarity_result=ticket.similarity_status_description,
        )
        send_signal(similarity_update, Ticket, ticket_id=ticket.id, status="found")

        ti_send_simchk_warning.apply_async(args=[id], countdown=5 * 60)

        notification_user_ids = set(
            chain(
                (ticket_creator.id,),
                ticket.reviews.values_list("reviewer", flat=True),
            )
        )
        self._create_notification_for_users(notification_user_ids, ticket)

    class CouldNotCreate(Exception):
        pass

    @staticmethod
    def _create_notification_for_users(user_ids: Set[int], ticket: "Ticket"):
        from codility.celery_tasks.notifications import send_notification
        from codility.notifications.public import PlagiarismDetectedNotification

        notifications_user_ids_that_can_see_personal_data = set(
            User.objects.filter(
                id__in=user_ids, can_see_candidate_personal_info=True
            ).values_list("id", flat=True)
        )
        notifications_user_ids_that_can_not_see_personal_data = user_ids.difference(
            notifications_user_ids_that_can_see_personal_data
        )

        if notifications_user_ids_that_can_see_personal_data:
            notification = PlagiarismDetectedNotification.create(
                user_ids=list(notifications_user_ids_that_can_see_personal_data),
                can_see_candidate_personal_info=True,
                ticket=ticket,
            )
            send_notification.delay(notification_data=notification.serialize())

        if notifications_user_ids_that_can_not_see_personal_data:
            notification = PlagiarismDetectedNotification.create(
                user_ids=list(notifications_user_ids_that_can_not_see_personal_data),
                can_see_candidate_personal_info=False,
                ticket=ticket,
            )
            send_notification.delay(notification_data=notification.serialize())


class TicketPermissionsMixin:
    def can_be_accessed_by(self, user: User, feature_mapper: FeatureMapper):
        return self.__accessible(user, feature_mapper)

    def can_be_cancelled_by(self, user: User, feature_mapper: FeatureMapper):
        return self.can_be_cancelled and self.__accessible(user, feature_mapper)

    def can_be_force_closed_by(self, user: User, feature_mapper: FeatureMapper):
        return self.status == "inuse" and self.__accessible(user, feature_mapper)

    def _can_be_reassessed_by(self, user: User, feature_mapper: FeatureMapper):
        return self.can_be_reassessed and self.__accessible(user, feature_mapper)

    def can_be_extended_by(self, user: User, feature_mapper: FeatureMapper):
        return self.can_be_extended and self.__accessible(user, feature_mapper)

    def can_be_reopened_as_codelive_by(self, user: User, feature_mapper: FeatureMapper):
        return (
            self.can_be_reopened_as_codelive
            and user.can_use_codelive()
            and self.can_be_accessed_by(user, feature_mapper)
        )

    def can_be_archived_by(self, user: User, feature_mapper: FeatureMapper):
        return self.__accessible(user, feature_mapper)

    def style_assessment_status(self, user: User, feature_mapper: FeatureMapper):
        def task_applies_for_style_assessment(task):
            return task.is_programming and not task.is_bugfixing

        if (
            not self.__accessible(user, feature_mapper)
            or not user.can_request_style_assessment
            or self.cancelled
            or all(
                not task_applies_for_style_assessment(tt.task)
                for tt in self.tickettasks
            )
        ):
            return "not-available"

        if hasattr(self, "styleassessmentrequest"):
            if self.styleassessmentrequest.is_done:
                return "done"
            else:
                return "in-progress"
        elif self.status != "closed":
            return "waits-for-submission"
        else:
            used_prg_langs = {
                tt.final_submit.task_context.get("prg_lang")
                for tt in self.tickettasks
                if tt.final_submit is not None
            }
            if used_prg_langs & set(settings.STYLE_ASSESSMENT_SUPPORTED_LANGS):
                return "available"
            else:
                return "not-supported-prg-lang"

    def can_report_be_shared_by(self, user: User, feature_mapper: FeatureMapper):
        return not self.cancelled and self.__accessible(user, feature_mapper)

    def can_report_be_exceptionally_viewed_by(self, user):
        # Even if the user cannot access the ticket, as an exception,
        # allow to open the report if the user is:
        # - a report recipient,
        # - or the CodeLive interviewer.
        return (
            Ticket.objects.visible(include_ghost=False).filter(id=self.id).exists()
            and user.is_recruiter
            and self.creator is not None
            and self.creator.is_recruiter
            and user.account == self.creator.account
            and (
                user.email in self.email_recipients
                or self.is_codelive
                and self.codelive_session.participants.filter(user=user).exists()
            )
        )

    def can_codelive_be_accessed_as_interviewer_by(self, user, session) -> bool:
        return is_codelive_interviewer(ticket=self, user=user, session=session)

    def can_codelive_scorecards_be_viewed_by(
        self, user: User, feature_mapper: FeatureMapper
    ) -> bool:
        return self.__accessible(user=user, feature_mapper=feature_mapper)

    def __accessible(self, user: User, feature_mapper: FeatureMapper) -> bool:
        if user.is_superuser:
            return True
        if self.creator is None or not self.creator.is_recruiter:
            return False
        return (
            Ticket.objects.visible_to(user=user, feature_mapper=feature_mapper)
            .filter(id=self.id)
            .exists()
        )


class Ticket(models.Model, TicketPermissionsMixin):
    ORIGIN_VALUES = (
        ("demo", "Demo"),
        ("cert", "Challenge"),
        ("template", "Template"),
        ("private", "Private"),
        ("public", "Public"),
        ("training", "Training"),
        ("dummy", "Dummy (used for load testing)"),
        ("try", "Trying a task or solution out, not billable"),
    )
    id = models.CharField(max_length=64, primary_key=True)
    campaign = models.ForeignKey(
        "campaigns.Campaign", on_delete=models.CASCADE, null=True, blank=True
    )
    exam = models.CharField(max_length=64, null=True, blank=True)
    identity = models.ForeignKey(
        Identity, on_delete=models.SET_NULL, null=True, blank=True
    )

    creator = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, null=True, blank=True)
    candidate = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="candidate_tickets",
    )
    create_date = models.DateTimeField(null=True, default=datetime.utcnow)
    intro_visit_date = models.DateTimeField(null=True, blank=True)
    start_date = models.DateTimeField(null=True, blank=True)
    close_date = models.DateTimeField(null=True, blank=True)

    # Some tasks should be triggered once the ticket is scored. This tells when the last tasks have been resolved.
    # If there's a result and this is blank, then some tasks still have to be resolved (or retried).
    notifications_finish_date = models.DateTimeField(null=True, blank=True)

    # Ticket result. We assume it's 'None' until the ticket has been fully evaluated.
    result = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    max_result = models.DecimalField(max_digits=12, decimal_places=2, null=True)
    modified_result_percent = models.IntegerField(null=True, blank=True)
    invoice = models.ForeignKey(
        Package, on_delete=models.CASCADE, null=True, blank=True
    )
    origin = models.CharField(max_length=12, null=False, choices=ORIGIN_VALUES)
    is_archived = models.BooleanField(default=False)
    candidate_country = models.CharField(max_length=10, null=True, blank=True)
    badge = models.CharField(max_length=15, null=True, blank=True)
    timelimit = models.PositiveIntegerField(
        null=True,
        blank=True,
        default=TICKET_DEFAULT_TIMELIMIT,
        help_text="Timelimit in seconds",
    )
    is_codelive = models.BooleanField(null=True, blank=True)
    external_id = models.CharField(
        max_length=128,
        null=True,
        blank=True,
        help_text="If ticket was created via API, this is an API-supplied identifier that "
        "would normally identify a candidate in an external system",
    )

    # "removed" is a ticket that was removed in the old way; it is
    # not visible to the user.
    removed = models.BooleanField(default=False)

    # "cancelled" is a ticket that was cancelled; it is
    # shown to the user, but only for archiving information.
    # A null value means False (not cancelled).
    cancelled = models.BooleanField(null=True, blank=True)

    # Ticket Inspection Reports ready
    # None -> No reports (either ticket not closed, or nothing has been found)
    # False -> generation _OR_ verification of ticket reports is pending
    # True -> some Ticket Inspection reports are available
    # We can assume that the majority of tickets will have NULL value in the database
    ti_rpts_ready = models.BooleanField(null=True, blank=True)

    demo_parent_ticket = models.ForeignKey(
        "Ticket", on_delete=models.CASCADE, null=True, blank=True
    )

    objects = TicketManager()

    # How many tasks can be in one ticket?
    # This is restricted by space for tabs in candidate UI.
    TASK_LIMIT = 10

    # Memoized lists of TicketTasks, Submits, and CodeSnapshots.
    # Reload the ticket to update.

    def get_creator(self) -> Union[User, "CustomAnonymousUser"]:
        if self.creator_id is not None:
            return self.creator

        if self.origin in ("training", "demo"):
            from codility.programmers.models import get_programmers_home_creator

            return get_programmers_home_creator()

        return CustomAnonymousUser()

    @property
    def nick(self):
        if not self.identity:
            return ""

        return self.identity.nick

    @property
    def email(self):
        return self.identity.email if self.identity else ""

    @property
    def first_name(self):
        return self.identity.first_name if self.identity else ""

    @property
    def last_name(self):
        return self.identity.last_name if self.identity else ""

    @property
    def phone(self):
        return self.identity.phone if self.identity else ""

    @property
    def is_english_only(self) -> bool:
        if self.creator:
            creator = self.creator
        elif self.is_paid_demo:
            creator = self.demo_parent_ticket.creator
        else:
            creator = None
        if creator and creator.is_recruiter:
            return creator.account.get_options().english_only
        else:
            return False

    def get_creator_account(self):
        creator_account = None
        if self.creator:
            creator_account = self.creator.account
        elif self.demo_parent_ticket and self.demo_parent_ticket.creator:
            creator_account = self.demo_parent_ticket.creator.account

        return creator_account

    def update_identity(self, *, save=True, **kwargs):
        unknown_keys = set(kwargs.keys()) - set(Identity.FIELDS)
        assert not unknown_keys, f"unknown identity fields: {unknown_keys}"

        creator_account = self.get_creator_account()

        identity_data = dict()
        for field in Identity.FIELDS:
            if self.identity:
                identity_data[field] = getattr(self.identity, field)
                if field in kwargs.keys():
                    identity_data[field] = kwargs.get(field) or ""
            else:
                identity_data[field] = kwargs.get(field) or ""

        candidate = Identity.find_candidate(
            account_id=creator_account.id if creator_account else None,
            email=identity_data["email"],
            first_name=identity_data["first_name"],
            last_name=identity_data["last_name"],
        )

        previous_identity = self.identity

        if candidate is not None:
            self.identity = candidate
            self.identity.phone = identity_data["phone"]
        else:
            if not identity_data.get("email", None):
                identity_data["origin_private_link"] = True
            self.identity = Identity.objects.create(
                **identity_data,
                account=creator_account,
            )

        if previous_identity is not None and self.identity != previous_identity:
            can_safely_delete_previous_identity = (
                Ticket.objects.filter(identity=previous_identity)
                .exclude(id=self.id)
                .count()
                == 0
            )

            if can_safely_delete_previous_identity:
                previous_identity.delete()

        self.identity.save()
        if save:
            self.save(update_fields=["identity"])

    def update_identity_from(self, ticket, save=True):
        if not ticket.identity:
            return
        kwargs = {k: getattr(ticket.identity, k) for k in Identity.FIELDS}
        self.update_identity(save=save, **kwargs)

    @property
    def final_result(self):
        if self.result is None:
            return None

        tickettasks = self.tickettasks
        total = 0
        for tt in tickettasks:
            tt_final_result = tt.final_result
            if tt_final_result is not None:
                total += tt_final_result

        return total

    @property
    def final_max_result(self):
        tickettasks = self.tickettasks
        max_result = 0
        for tt in tickettasks:
            max_result += tt.final_max_result
        return max_result

    @property
    def final_result_pr(self):
        if self.result is None:
            return None

        if self.final_max_result == 0:
            return 0

        return float(100 * self.final_result) / float(self.final_max_result)

    @property
    def is_result_modified(self):
        return any(
            [
                tt.modified_result is not None and (not tt.task.no_assessment)
                for tt in self.tickettasks
            ]
        )

    @property
    def created_by(self):
        if self.creator:
            return self.creator.username
        # then origin can't be 'private', 'template', 'public' or 'try'
        return self.origin

    @property
    def created_by_email(self):
        if self.creator:
            return self.creator.email
        # then origin can't be 'private', 'template', 'public' or 'try'
        return self.origin

    def reset(self):
        """Reset all memoized properties."""
        reset_memo(self)

    @memo
    def tickettasks(self) -> List["TicketTask"]:
        # Avoid filtering over the tickettask_set here, it breaks prefetch_related
        return sorted(self.tickettask_set.all(), key=lambda x: x.num)

    @memo
    def has_multi_file_tasks(self) -> bool:
        return any(ticket_task.task.is_multi_file for ticket_task in self.tickettasks)

    @memo
    def has_weighted_scores(self) -> bool:
        return any(self.task_weights)

    @memo
    def task_weights(self) -> List[Optional["TaskWeight"]]:
        return [ticket_task.task_weight for ticket_task in self.tickettasks]

    @memo
    def time_used(self):
        """Total time used by the candidate, in seconds"""
        if self.close_date and self.start_date:
            return int((self.close_date - self.start_date).total_seconds())
        return None

    @property
    def time_used_min(self):
        return to_minutes(self.time_used)

    @memo
    def total_correctness(self):
        """Compute total correctness (trac #2283).
        It is defined if there's at least one non-N/A"""
        # text task is not included to compute average
        tts = [x for x in self.tickettasks if not x.task.no_assessment]
        if all(tt.correctness is None for tt in tts):
            return None
        else:
            return sum(tt.correctness or 0 for tt in tts) / len(tts)

    @memo
    def total_performance(self):
        """Compute total performance (trac #2283).
        It is defined if there are no N/A's"""
        # text task is not included to compute average
        tts = [x for x in self.tickettasks if not x.task.no_assessment]
        if len(tts) == 0 or any(tt.performance is None for tt in tts):
            return None
        else:
            return sum(tt.performance for tt in tts) / len(tts)

    @memo
    def submits(self):
        return sorted(self.submit_set.all(), key=lambda x: x.submit_date)

    @memo
    def solutions(self) -> List["CodeSnapshot"]:
        sols = list(self.codesnapshot_set.all().order_by("timestamp"))
        # Add CodeSnapshots to submits that don't have them
        submit_ids = set(sol.submit.id for sol in sols if sol.submit)
        added = False
        for submit in self.submits:
            if submit.id not in submit_ids:
                new_code = submit.solution or ""
                logger.warning(
                    "[LEGACY] Accessing 'solutions' property-method and creating new CodeSnapshot",
                    stack_info=True,
                )
                CodeSnapshot.objects.create(
                    code=new_code,
                    _task_context=submit.task_context,
                    timestamp=submit.submit_date,
                    ticket=self,
                    task=Task.objects.get(name=submit.task),
                    submit=(
                        submit
                        if submit.mode == "verify" or submit.mode == "final"
                        else None
                    ),
                )
                added = True

        if added:
            # Archive whole ticket in Evaluations API.
            # This is acceptable because this feature only is ever triggered by old submits created
            # before the introduction of CodeSnapshots.

            # Import here to avoid circular imports.
            from codility.tickets.solutions_archiver import SolutionsArchiver

            archiver = SolutionsArchiver(write_only=False)
            for tt in self.tickettasks:
                archiver.archive_ticket_task(tt)
            # We added a snapshot, reload them all again
            sols = list(self.codesnapshot_set.all().order_by("timestamp"))
        return sols

    @property
    def solutions_except_submits(self):
        return [solution for solution in self.solutions if solution.submit_id is None]

    # candidate_data (one-to-one)

    def get_candidate_url(self):
        if not (self.origin in ["private", "public"]):
            return {
                "value": None,
                "tooltip": "Candidate Profile is created only for private and public tickets",
            }

        identity = self.identity

        if identity is None:
            return {
                "value": None,
                "tooltip": "Candidate Profile could not be created for anonymized candidate",
            }

        if identity.origin_private_link:
            return {
                "value": None,
                "tooltip": "Candidate Profile could not be created due to lack of candidate email",
            }

        return {"value": reverse("candidate_profile", args=[identity.id])}

    def get_candidate_data(self):
        """Return candidate_data, or None if doesn't exist"""
        try:
            return self.candidate_data
        except ObjectDoesNotExist:
            return None

    # data_requirements (one-to-one with campaigns)

    def get_data_requirements(self):
        """Returns the ticket's data requirements, or None if not applicable."""
        try:
            return self.data_requirements
        except ObjectDoesNotExist:
            return None

    def get_mail_status(self):
        try:
            mail_status = TicketMailStatus.objects.get(ticket=self)
            return mail_status.status
        except TicketMailStatus.DoesNotExist:
            return None

    @property
    def result_int(self):
        return get_result_int(self.result)

    @property
    def max_result_int(self):
        if self.max_result is not None:
            return int(round(self.max_result))
        return None

    @memo
    def _option_map(self):
        return {option.opt: option.data for option in self.options.all()}

    def _has_option(self, opt_name):
        return opt_name in self._option_map

    def _get_option(self, opt_name, default_value=None):
        return self._option_map.get(opt_name, default_value)

    def _set_option(self, opt_name, value):
        option, create = self.options.get_or_create(opt=opt_name)
        option.data = value
        option.save()
        del self._option_map

    def _remove_option(self, opt_name):
        try:
            option = self.options.get(opt=opt_name)
            option.delete()
            del self._option_map
        except ObjectDoesNotExist:
            pass

    def create_options(self, options):
        self.options.bulk_create(
            [
                TicketOption(
                    opt=opt_name,
                    data=getattr(Ticket, opt_name).python_to_db(opt_data),
                    ticket=self,
                )
                for opt_name, opt_data in options.items()
            ]
        )
        del self._option_map

    @property
    def current_task(self) -> Task:
        tt = self.current_tickettask
        return tt.task

    @property
    def current_tickettask(self) -> "TicketTask":
        """
        Make a best guess at which task could be considered current
        for the candidate.
        """

        zero_date = datetime.fromtimestamp(0)

        def tt_key(tt):
            last_activity = max(
                tt.latest_access_date or zero_date,
                tt.start_date or zero_date,
                tt.close_date or zero_date,
                tt.solutions[-1].timestamp if tt.solutions else zero_date,
            )

            return (
                # Prefer open tasks, then new, then closed
                {"open": 3, "new": 2, "closed": 1}[tt.status],
                # Select task that was last active
                last_activity,
                # Prefer tasks with lower number in the ticket
                -tt.num,
            )

        return max(self.tickettasks, key=tt_key)

    def all_prg_langs_exts(self) -> List[str]:
        """returns all programming languages possible to use in this test (in any way)"""
        return sorted(
            {
                lang
                for tt in self.tickettask_set.prefetch_related("task")
                for lang in get_prg_langs_from_tickettask(tt)
            }
        )

    @memo
    def pasted_codes_dict(self):
        res = dict([(p.result_snapshot.id, p) for p in self.pastedcode_set.all()])
        return res

    def pasted_codes(self, cs):
        res = self.pasted_codes_dict.get(cs.id)
        if res is None:
            return []
        else:
            return [res]

    def save_pasted_code(
        self, cs: "CodeSnapshot", start_pos: int, end_pos: int
    ) -> None:
        assert (
            start_pos is not None and end_pos is not None
        ), "Missing start_pos/end_pos in pasted code"

        if not (0 <= start_pos and start_pos <= end_pos and end_pos <= len(cs.code)):
            logger.warning(
                "Invalid pasted code parameters, ticket id=%s, start_pos=%d, end_pos=%d, code_len=%d",
                self.id,
                start_pos,
                end_pos,
                len(cs.code),
            )

        # normalize start_pos / end_pos
        start_pos = min(max(0, start_pos), len(cs.code))
        end_pos = min(max(start_pos, end_pos), len(cs.code))
        assert (
            0 <= start_pos and start_pos <= end_pos and end_pos <= len(cs.code)
        ), "Invalid start_pos/end_pos"

        start_line = cs.code[0:start_pos].count("\n") + 1
        end_line = cs.code[0:end_pos].rstrip().count("\n") + 1
        PastedCode.objects.create(
            task=cs.task,
            ticket=self,
            pasted=cs.code[start_pos:end_pos],
            start_pos=start_pos,
            end_pos=end_pos,
            start_line=start_line,
            end_line=end_line,
            result_snapshot=cs,
        )

    @memo
    def get_tracking_info(self):
        """returns intepreted tracking info"""
        if self.trackers is None:
            return {}

        def convert_data_to_timestamps(t_key, data_filter=None):
            r = []
            if t_key not in self.trackers:
                return r
            t = self.trackers[t_key]
            start_time, interval, data = t["start_time"], t["interval"], t["data"]
            pairs = sorted(
                [
                    (int(k), v)
                    for k, v in data.items()
                    if data_filter is None or data_filter(v)
                ]
            )
            for k, v in pairs:
                r.append(
                    {
                        "start_time": start_time + k * interval,
                        "end_time": start_time + (k + 1) * interval,
                        "value": v,
                    }
                )
            return r

        def compress_intervals(r):
            n = len(r)
            if n == 0:
                return []
            rr = []
            i0 = 0
            for i, row in enumerate(r):
                if i == n - 1 or row["end_time"] != r[i + 1]["start_time"]:
                    rr.append(
                        {
                            "start_time": r[i0]["start_time"],
                            "end_time": row["end_time"],
                            "value": r[i0]["value"],
                        }
                    )
                    i0 = i + 1
            return rr

        res = {
            "focus_inactive": convert_data_to_timestamps("focus", lambda x: x == 0),
            "focus_active": convert_data_to_timestamps("focus", lambda x: x == 1),
            "keypress": sorted(
                [
                    [r["start_time"], r["value"]]
                    for r in convert_data_to_timestamps("keypress")
                ]
            ),
        }
        for k in ["focus_active", "focus_inactive"]:
            res[k] = compress_intervals(res[k])
        # keypresses, but the timestamp are given in milliseconds since EPOCH
        res["keypress_ms"] = [[t * 1000, v] for t, v in res["keypress"]]
        return res

    def _adjust_tracker_data(
        self,
        data,
        data_interval,
        new_interval,
        data_start_time,
        new_start_time,
        reduce_op,
    ):
        res = {}
        for t, v in data.items():
            t = int(t)
            tt = (
                (t * data_interval) + (data_start_time - new_start_time)
            ) // new_interval
            if tt in res:
                res[tt] = reduce_op(res[tt], v)
            else:
                res[tt] = v
        return res

    def save_trackers_info(self, trackers_info):
        DEFAULT_TRACKER_INTERVAL = 60
        t = self.trackers or {}
        ticket_start_time = dt_to_timestamp(self.start_date)
        for name, tmp in trackers_info.items():
            if isinstance(tmp, dict):
                row = tmp
            else:
                (data, interval) = json.loads(tmp)
                row = {"data": data, "interval": interval}
            if name not in t:
                t[name] = {
                    "start_time": ticket_start_time,
                    "data": {},
                    "interval": DEFAULT_TRACKER_INTERVAL,
                }

            if name in ["focus"]:

                def reduce_op(x, y):
                    return x or y

            else:

                def reduce_op(x, y):
                    return x + y

            data = self._adjust_tracker_data(
                row["data"],
                row["interval"],
                t[name]["interval"],
                ticket_start_time,
                t[name]["start_time"],
                reduce_op,
            )

            # print "name=%s data after adjust=%s" % (name, data)
            merged_data = dict([(int(k), v) for k, v in t[name]["data"].items()])
            for k, v in data.items():
                if k in merged_data:
                    merged_data[k] = reduce_op(merged_data[k], v)
                else:
                    merged_data[k] = v
            t[name]["data"] = merged_data

        self.trackers = t
        self.save()

    @staticmethod
    def _regenerate_presigned_url_and_upload_solution(
        evaluations_solution_s3_client: EvaluationsSolutionS3Client,
        evaluation_info: "EvaluationInfo",
        solution_files_obj: SolutionFiles,
    ) -> str:
        evaluations_api_client = EvaluationsApiClientHttp(
            base_url=settings.EVALUATIONS_API_BASE_URL,
            api_key=settings.EVALUATIONS_API_API_KEY,
        )
        # When pregined URL is expired, regenerate it and upload the solution again
        presigned_url = evaluations_api_client.regenerate_presigned_url(
            evaluation_id=evaluation_info.evaluation_id,
        )

        # Update EvaluationInfo model with new presigned URL
        evaluation_info.solution_tar_gz_presigned_url = presigned_url
        evaluation_info.save()

        s3_object_version_id = evaluations_solution_s3_client.upload_solution(
            solution_presigned_url=evaluation_info.solution_tar_gz_presigned_url,
            solution_files=solution_files_obj,
        )

        return s3_object_version_id

    def save_solution(
        self,
        task_name: str,
        solution: Optional[str] = None,
        solution_files: Optional[Dict[str, str]] = None,
        *,
        task_context: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        submit: Optional["Submit"] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "CodeSnapshot":
        """
        Saves a solution for a given task along with its context, submission details, and additional information if specified in `extra`
        like "PastedCode" entries. Returns the stored "CodeSnapshot" instance.

        Parameters:
        - task_name (str): The name of the task for which the solution is being saved.
        - solution (Optional[str]): The solution code or text to be saved. If not provided, `solution_files` should be provided.
        - solution_files (Optional[Dict[str, str], optional): A dictionary containing the file path and file content of the solution
        files to be saved in a tar.gz file in S3. If not provided, `solution` should be provided.
        - task_context (Dict[str, Any]): The Task context of the solution being saved (e.g. specifies programming language used).
        - timestamp (Optional[datetime], optional): The timestamp when the solution was created or submitted. Defaults to the current UTC time if not provided.
        - submit (Optional[Submit], optional): An instance of the Submit model indicating the submission details. Defaults to None.
        - extra (Optional[Dict[str, Any]], optional): Additional data to be saved with the solution, which can include:
            - "paste_start": The start position of a code paste (int).
            - "paste_end": The end position of a code paste (int).
        - **kwargs (Any): Additional keyword arguments to be passed to the CodeSnapshot creation.

        Raises:
        - Exception: If the ticket (assumed context of this method) is already closed.
        or if both solution and solution_files are provided.

        Returns:
        - CodeSnapshot: An instance of the created CodeSnapshot object after saving the solution.

        Note: This method also logs an error and raises an exception if the ticket is closed.
        """
        if extra is None:
            extra = {}
        if self.is_closed:
            logger.error(
                "save_solution: ticket is already closed, %s %s %s, %s",
                self.id,
                task_name,
                json.dumps(task_context),
                ", ".join(traceback.format_stack()),
            )
            raise Exception("Ticket is already closed")
        if not (solution is None or solution_files is None):
            raise Exception(
                "Both solution and solution_files cannot be provided together."
            )
        if solution is None and solution_files is None:
            raise Exception("Either solution or solution_files should be provided.")

        timestamp = timestamp or datetime.utcnow()
        task = Task.objects.get(name=task_name)
        task.validate_cleaned_task_context(task_context)

        if solution is not None:
            code_snapshot = CodeSnapshot.objects.create(
                task=task,
                ticket=self,
                code=solution,
                _task_context=task_context,
                submit=submit,
                timestamp=timestamp,
                **kwargs,
            )

            if "paste_start" in extra and "paste_end" in extra:
                self.save_pasted_code(
                    code_snapshot,
                    int(extra.get("paste_start")),
                    int(extra.get("paste_end")),
                )
            self.reset()  # Reset all memoized properties
            return code_snapshot

        # If solution_files are provided, save them to S3 and get the solution path
        # We need to fetch the corresponding EvaluationInfo object to get the presigned URL
        try:
            evaluation_info = EvaluationInfo.objects.only(
                "solution_tar_gz_presigned_url"
            ).get(task_name=task.name, ticket=self)
        except EvaluationInfo.DoesNotExist:
            raise Exception(
                f"Could not find EvaluationInfo for task {task.name} and ticket {self.id}."
            )

        evaluations_solution_s3_client = EvaluationsSolutionS3Client()

        solution_files_obj = SolutionFiles.from_file_dict(solution_files)

        try:
            s3_object_version_id = evaluations_solution_s3_client.upload_solution(
                solution_presigned_url=evaluation_info.solution_tar_gz_presigned_url,
                solution_files=solution_files_obj,
            )
        except PresignedUrlExpiredError:
            s3_object_version_id = self._regenerate_presigned_url_and_upload_solution(
                evaluations_solution_s3_client=evaluations_solution_s3_client,
                evaluation_info=evaluation_info,
                solution_files_obj=solution_files_obj,
            )

        return CodeSnapshot.objects.create(
            task=task,
            ticket=self,
            code=None,  # code is not saved in the database for solution_files
            _task_context=change_context_from_frontend_to_task_server_format(
                task_context=task_context, task_name=task.name
            ),
            submit=submit,
            timestamp=timestamp,
            solution_tar_gz_version_id=s3_object_version_id,  # save the version id of the tar.gz file in S3
            **kwargs,
        )

    def user_resigned(self, resign_time=None):
        opt_id = "was_resignation"
        resign_time = resign_time or datetime.utcnow()
        self._set_option(opt_id, dt_to_timestamp(resign_time))

    def _create_submit(
        self,
        mode,
        task_name,
        solution,
        *,
        task_context,
        test_data=None,
        submit_date=None,
        timestamp=None,
        extra_options=None,
        file_submit=None,
        solution_path: Optional[str] = None,
        solution_tar_gz_version_id: Optional[str] = None,
        evaluation_info_id: Optional[str] = None,
    ):
        """
        Create a submit with associated CodeSnapshot. Note that `submit_date`
        should be real, while `timestamp` will be used to put the submit on a
        timeline.
        FOR INTERNAL USE ONLY, use create_submit for normal use. It adds sanity checks.
        """
        test_data = test_data or {}
        extra_options = extra_options or {}

        options = parse_options_string(test_data, extra_options)
        submit_date = submit_date or datetime.utcnow()
        timestamp = timestamp or submit_date
        lexer_lang = get_task_highlight(task_name, task_context)
        fingerprint_of_solution = None
        if mode == "final":
            from codility.utils.solution_fingerprint import solution_fingerprint

            fingerprint_of_solution = solution_fingerprint(
                solution, lexer_lang
            )  # TODO; change prg_lang to context here
        if self.origin == "dummy":
            raise NotImplementedError("Dummy tickets are no longer supported.")

        submit = Submit.objects.create(
            mode=mode,
            ticket=self,
            task=task_name,
            solution=solution,
            submit_date=submit_date,
            eval_date=None,
            solution_fingerprint=fingerprint_of_solution,
            options=options,
            file_submit=file_submit,
            _task_context=task_context,
            solution_path=solution_path,
            solution_tar_gz_version_id=solution_tar_gz_version_id,
            evaluation_info_id=evaluation_info_id,
        )
        self.save_solution(
            task_name=task_name,
            solution=solution,
            timestamp=timestamp,
            submit=submit,
            task_context=task_context,
        )

        return submit

    def create_submit(
        self,
        mode,
        task_name,
        solution,
        *,
        task_context,
        test_data=None,
        submit_date=None,
        timestamp=None,
        extra_options=None,
        file_submit=None,
        solution_path: Optional[str] = None,
    ):
        """
        Create a submit with associated CodeSnapshot. Note that `submit_date`
        should be real, while `timestamp` will be used to put the submit on a
        timeline.
        """

        if self.is_closed:
            raise Exception("Ticket is already closed")
        tickettask: Optional[TicketTask] = self.task_of_ticket(task_name)
        if tickettask is None:
            raise Exception("Ticket does not have this task")
        elif tickettask.status == "closed":
            raise Exception("Cannot create a submission for a closed task")

        validate_task_context_for_ticket_task(task_context, tickettask)

        return self._create_submit(
            mode,
            task_name,
            solution,
            task_context=task_context,
            test_data=test_data,
            submit_date=submit_date,
            timestamp=timestamp,
            extra_options=extra_options,
            file_submit=file_submit,
            solution_path=solution_path,
        )

    def timelimit_min(self):
        if self.timelimit is None:
            return None
        return (self.timelimit + 59) // 60

    @property
    def expected_close_date(self):
        if not self.start_date:
            return None
        if self.timelimit is None:
            return None
        return self.start_date + timedelta(seconds=self.timelimit)

    @property
    def after_expected_close_date(self):
        if self.expected_close_date is None:
            return False
        return self.expected_close_date + timedelta(seconds=60) < datetime.utcnow()

    @property
    def time_remaining_sec(self):
        "Real time candidate has to finish the ticket"
        if not self.start_date:
            if self.sharp_end_time is None:
                return self.timelimit
            else:
                diff = self.sharp_end_time - datetime.utcnow()
                if self.timelimit is None:
                    return max(0, int(diff.total_seconds()))
                return min(self.timelimit, max(0, int(diff.total_seconds())))
        elif self.is_closed:
            return 0
        elif self.expected_close_date is None:
            return None
        else:
            diff = self.expected_close_date - datetime.utcnow()
            return max(0, int(diff.total_seconds()))

    @property
    def time_elapsed_sec(self):
        if self.start_date is None:
            return None
        diff = datetime.utcnow() - self.start_date
        return max(0, diff.total_seconds())

    def time_remaining_min(self):
        res = self.time_remaining_sec
        if res is None:
            return None
        else:
            return max(0, (res + 59) // 60)

    def seconds_to_start(self):
        valid_from = self.valid_from
        if valid_from is None:
            return None
        return int((valid_from - datetime.utcnow()).total_seconds())

    @property
    def is_valid_to_start(self):
        "safe check if ticket can be started"
        now = datetime.utcnow().replace(tzinfo=None)
        return self.valid_to_start_begin_time <= now < self.valid_to_start_end_time

    @property
    def is_closed(self):
        return self.close_date is not None

    @property
    def valid_to_start_begin_time(self):
        if self.valid_from:
            return self.valid_from
        return datetime.min

    @property
    def valid_to_start_end_time(self):
        dates = [self.valid_to, self.sharp_end_time, datetime.max]
        return min(list(filter(bool, dates)))

    status = QuerySetAnnotation("with_status")
    can_be_cancelled = QuerySetAnnotation("with_possible_actions")
    can_be_extended = QuerySetAnnotation("with_possible_actions")
    can_be_reassessed = QuerySetAnnotation("with_possible_actions")
    can_be_reopened_as_codelive = QuerySetAnnotation("with_possible_actions")

    def status_text(self):
        """Human-readable status."""
        st = self.status
        if st == "cancelled":
            return "cancelled"
        if st == "new":
            if self.extended:
                return "not re-started"
            elif self.is_public:
                return "ghost"
            else:
                return "not started"
        elif st == "inuse":
            r = self.time_remaining_min()
            if r is None:
                return "started"
            elif r > 0:
                return "started, %d min. remaining" % r
            else:
                return "started, timed out (waiting for evaluation)"
        elif st == "ineval":
            return "completed, waiting for evaluation"
        else:
            return "completed"

    @property
    def status_raw(self):
        return self.status

    @property
    def can_resend_invitation(self):
        return (self.status == "new") and self.email

    @property
    def eval_date(self):
        cursor = custom_sql(
            "SELECT MAX(eval_date) FROM submits WHERE ticket=%s", [self.id]
        )
        return cursor.fetchone()[0]

    @property
    def unfinished_tasks(self):
        res = []
        for tt in self.tickettasks:
            if tt.close_date is None:
                res.append(tt.task.name)
        return res

    @property
    def unfinished_tasks_count(self):
        return len(self.unfinished_tasks)

    @property
    def tasks_count(self):
        return self.tickettask_set.count()

    def task_of_ticket(self, task_name) -> Optional["TicketTask"]:
        for ticket_task in self.tickettask_set.all():
            if ticket_task.task.name == task_name:
                return ticket_task
        return None

    def ticket_task_id(self, task_name):
        ticket_task = self.task_of_ticket(task_name)
        if ticket_task:
            return ticket_task.id
        return None

    def get_task_names(self):
        return [tt.task.name for tt in (self.tickettask_set.all().order_by("num"))]

    def get_task_scores_with_ids(self):
        task_scores_with_ids = {}
        for tt in self.tickettask_set.all().order_by("num"):
            task_scores_with_ids[tt.id] = tt.result_percent
        return task_scores_with_ids

    @memo
    def task_infos(self) -> List[TicketTaskInfo]:
        return [tt.task_info for tt in self.tickettask_set.all().order_by("num")]

    @staticmethod
    def _get_used_languages(submits, final_submits):
        """
        Select the submitted languages, sorted by popularity in submits.
        Count only final submits, unless there are none - then fall back to all submits.

        Return as list of strings
        """
        import collections

        relevant_submits = [
            submit
            for submit in (final_submits or submits)
            if "prg_lang" or "technology" in submit.task_context
        ]
        all_languages = [
            submit.task_context.get("prg_lang") or submit.task_context.get("technology")
            for submit in relevant_submits
        ]
        return [
            lang
            for (lang, __) in collections.Counter(
                all_languages
            ).most_common()  # returns all elements in decreasing order from most used one.
        ]

    @staticmethod
    def _get_used_languages_names(submits, final_submits):
        import collections

        relevant_submits = [
            submit
            for submit in (final_submits or submits)
            if "prg_lang" or "technology" in submit.task_context
        ]
        all_languages = [
            submit.tickettask.task.display_prg_lang(
                submit.task_context.get("prg_lang")
                or submit.task_context.get("technology")
            ).name
            for submit in relevant_submits
        ]
        languages = [
            lang
            for (lang, __) in collections.Counter(
                all_languages
            ).most_common()  # returns all elements in decreasing order from most used one.
        ]

        if not languages:
            languages = ["(None)"]
        return languages

    class TooManyTasks(Exception):
        pass

    def replace_tasks(self, task_infos):
        from codility.tickets.utils.ticket_builder import tickettask_from_task_info

        task_infos = list(task_infos)

        # Codelive template task is not counted as a 'proper' task,
        # it is automatically added to the tickets started as CodeLive
        # so tests with 6 tasks (TASK_LIMIT) get one more above the limit
        tasks_number = len(
            [
                ti
                for ti in task_infos
                if ti.get("name", "")
                not in {
                    CODELIVE_WHITEBOARD_TEMPLATE_TASK,
                }
            ]
        )

        if tasks_number > self.TASK_LIMIT:
            raise Ticket.TooManyTasks()
        assert len(task_infos) == len(
            {task_info["name"] for task_info in task_infos}
        ), "replace_tasks: no duplicates allowed"

        self.tickettask_set.all().delete()

        tickettasks = TicketTask.objects.bulk_create(
            tickettask_from_task_info(self.id, num, task_info)
            for (num, task_info) in enumerate(task_infos)
        )

        self.max_result = sum(tt.max_result for tt in tickettasks)
        self.save(update_fields=["max_result"])
        self.reset()  # reset self.tickettasks

    @property
    def cert_badge(self):
        if not hasattr(self, "_cert_badge"):
            badge = None
            for tt in self.tickettasks:
                s = tt.final_submit
                if s is None:
                    continue
                if s.cert_badge is None:
                    badge = None
                    break
                if badge is None:
                    badge = s.cert_badge
                else:
                    if s.cert_badge == "silver":
                        badge = s.cert_badge
            self._cert_badge = badge
        return self._cert_badge

    def _badge_to_desc(self, badge):
        if badge == "golden":
            return "correct functionality and scalability"
        elif badge == "silver":
            return "correct functionality, problems with scalability"
        elif badge == "golden-no-extreme":
            return "solution functionality and scalability is almost ok, it misses some extreme cases"
        elif badge == "silver-no-extreme":
            return "solution functionality is almost ok, it misses some extreme cases, also there are problems with scalability"
        else:
            return None

    @property
    def cert_badge_desc(self):
        return self._badge_to_desc(self.cert_badge)

    def solution_desc(self, task_name, force_save=True):
        return self._get_option("custom_sol_desc.%s" % task_name)

    @property
    def result_pr(self):
        # TODO: replace uses of result_pr|floatformat:"0" with result_pr_int
        r = self.result
        if r is None:
            return None
        m = self.max_result
        if m is None or m <= 0:
            m = 100
        return float(100 * r) / float(m)

    @property
    def result_pr_int(self):
        "Percentage result canonically rounded to nearest integer."
        return round_result(self.result_pr)

    @property
    def final_result_pr_int(self):
        "Percentage result canonically rounded to nearest integer."
        return round_result(self.final_result_pr)

    @property
    def is_template(self):
        """Return if a ticket is a template (campaign template ticket)"""
        return self.origin == "template"

    @property
    def is_public(self):
        """Return if a ticket is public (i.e. created via honeypot) or not"""
        return self.origin == "public"

    @property
    def is_private(self):
        """Return if a ticket is private (i.e. not created via honeypot) or not"""
        return self.origin == "private"

    @property
    def is_demo(self):
        """Return if a ticket is demo (paid or not)"""
        return self.origin == "demo"

    @property
    def is_paid_demo(self):
        return self.origin == "demo" and self.demo_parent_ticket is not None

    @property
    def is_cert(self):
        """Return if a ticket is a certificate ticket"""
        return self.origin == "cert"

    @property
    def is_training(self):
        return self.origin == "training"

    @property
    def is_try(self):
        return self.origin == "try"

    @property
    def is_billable(self):
        """Return if the ticket should be counted in the subscription plan"""
        return self.origin in ["public", "private"] and self.creator is not None

    @property
    def demo_url(self):
        if self.demo_task:
            ret = reverse(
                "take_custom_prepare_test",
                kwargs={"demo_name": self.demo_task, "ticket_id": self.id},
            )
        else:
            ret = reverse("take_sample_prepare_test", kwargs={"ticket_id": self.id})
        return ret

    def any_others_in_evaluation(self):
        """Check if any other tickets with earlier close_date are evaluating right now"""

        if not self.close_date:
            raise Exception("Ticket has to have a close date")

        how_many = (
            Ticket.objects.cert_tickets()
            .filter(exam=self.exam, close_date__lt=self.close_date, result__isnull=True)
            .exclude(identity__email__contains="@codility")
            .count()
        )

        return how_many > 0

    def which_granted(self):
        # how many certificates have been granted this month
        if self._get_option("which_cert_granted"):
            return self._get_option("which_cert_granted")
        else:
            if self.badge not in ["silver", "golden"]:
                return None

            # need to be checked because we can't know the place for candidate if there are other solutions (which have earlier close date) is evaluating
            if self.any_others_in_evaluation():
                return None

            if self.close_date:
                t = self.close_date
            else:
                t = datetime.utcnow()

            # Don't count multiple submissions from the same email (Trac #2698)
            how_many = len(
                set(
                    Ticket.objects.cert_tickets()
                    .filter(exam=self.exam, close_date__lt=t, badge=self.badge)
                    .exclude(identity__email__contains="@codility")
                    .exclude(identity__email=self.email)
                    .values_list("identity__email", flat=True)
                )
            )

            if how_many < 10 or how_many >= 20:
                if how_many % 10 == 0:
                    result = str(how_many + 1) + "st"
                elif how_many % 10 == 1:
                    result = str(how_many + 1) + "nd"
                elif how_many % 10 == 2:
                    result = str(how_many + 1) + "rd"
                else:
                    result = str(how_many + 1) + "th"
            else:
                result = str(how_many + 1) + "th"

            self._set_option("which_cert_granted", result)
            #            self.save()

            return result

    def access_data(
        self,
        ip,
        http_user_agent=None,
        http_accept=None,
        http_accept_encoding=None,
        http_accept_language=None,
        geoip_country_code=None,
    ):
        """Log the IP if it doesn't match the one from which the ticket has been started"""
        assert ip
        TicketAccess.objects.get_or_create(
            ticket=self,
            ip=ip,
            http_user_agent=http_user_agent or "",
            http_accept=http_accept or "",
            http_accept_encoding=http_accept_encoding or "",
            http_accept_language=http_accept_language or "",
            geoip_country_code=geoip_country_code or "",
        )

    class Event:
        """A single event related to a ticket."""

        def __init__(self, timestamp, kind, task_name=None, extra_desc=None):
            # Lose microsecond accuracy - ticket options are accurate
            # only down to 1 second, and this screws with the timeline
            # ordering.
            self.timestamp = timestamp.replace(microsecond=0)
            self.kind = kind
            self.task_name = task_name
            self.extra_desc = extra_desc

        def sort_key(self):
            n = {
                "start": 0,
                "submit": 5,
                "verify": 5,
                "final": 5,
                "end_task": 10,
                "save": 20,
                "save_with_paste": 20,
                "start_task": 50,
                # "Switch" events are often inserted after other
                # events, but with the same timestamp.
                "switch": 60,
                "resign": 90,
                "end": 100,
            }[self.kind]
            return (self.timestamp, n, self.task_name)

        @property
        def description(self):
            if self.kind == "start":
                desc = "Opened test"
            elif self.kind == "end":
                desc = "Closed test"
            elif self.kind == "resign":
                desc = "User resigned from solving further problems"
            elif self.kind == "start_task":
                desc = "Opened task %s" % self.task_name
            elif self.kind == "end_task":
                desc = "Closed task %s" % self.task_name
            elif self.kind == "save":
                desc = "Autosave for task %s" % self.task_name
            elif self.kind == "save_with_paste":
                desc = "Autosave for task %s (with pasted lines)" % self.task_name
            elif self.kind == "final":
                desc = "Final version of task %s" % self.task_name
            elif self.kind == "verify":
                desc = "Verify request for task %s" % self.task_name
            elif self.kind == "submit":
                desc = "Submission for task %s" % self.task_name
            elif self.kind == "switch":
                desc = "Switched to task %s" % self.task_name
            else:
                raise Exception("invalid kind %s" % self.kind)

            if self.extra_desc:
                desc = desc + " -- " + self.extra_desc

            return desc

        def __repr__(self):
            return repr((self.kind, self.task_name))

    def get_events(self, filtered):
        events = set()

        # Gather all the events related to a task

        if self.start_date:
            events.add(Ticket.Event(self.start_date, "start"))
        if self.close_date:
            events.add(Ticket.Event(self.close_date, "end"))

        for option in self.options.filter(opt__startswith="start."):
            _, _, task_name = option.opt.partition(".")
            t = timestamp_to_dt(option.data)
            events.add(Ticket.Event(t, "start_task", task_name))

        for option in self.options.filter(opt__startswith="end."):
            _, _, task_name = option.opt.partition(".")
            t = timestamp_to_dt(option.data)
            events.add(Ticket.Event(t, "end_task", task_name))

        for sol in self.solutions:
            extra_desc = None
            if sol.submit:
                if sol.submit.mode == "final":
                    kind = "final"
                elif sol.submit.mode == "verify":
                    kind = "verify"
                    if sol.submit.result == 100:
                        extra_desc = "SUCCESS"
                    else:
                        extra_desc = "FAIL"
                        vr = sol.submit.verification_result
                        if vr is not None:
                            extra_desc += " (%s)" % vr
                else:
                    kind = "submit"
                    extra_desc = "mode: " + sol.submit.mode
            else:
                if sol.has_pasted_lines:
                    kind = "save_with_paste"
                else:
                    kind = "save"
            events.add(Ticket.Event(sol.timestamp, kind, sol.task.name, extra_desc))

        if self.was_resignation:
            events.add(Ticket.Event(timestamp_to_dt(self.was_resignation), "resign"))

        def sorted_events():
            return sorted(events, key=Ticket.Event.sort_key)

        if not filtered:
            return sorted_events()

        # Add "switch" events
        for ev1, ev2 in pairwise(sorted_events()):
            if (
                ev1.task_name != ev2.task_name
                and ev2.task_name
                and ev2.kind != "start_task"
            ):
                events.add(Ticket.Event(ev1.timestamp, "switch", ev2.task_name))

        # Remove autosaves
        for ev in list(events):
            if ev.kind == "save":
                events.remove(ev)

        # Remove superfluous "switch" events
        for ev1, ev2 in pairwise(sorted_events()):
            if ev2.kind == "switch" and ev1.task_name == ev2.task_name:
                events.remove(ev2)

        return sorted_events()

    @memo
    def events_list(self):
        return self.get_events(filtered=True)

    @memo
    def full_events_list(self):
        return self.get_events(filtered=False)

    @property
    def test_url(self):
        assert self.origin != "template"
        # TODO consider TicketManager.visible
        safe_id = self.id  # TODO add escaping
        url = get_site_url() + reverse("test", args=[safe_id])
        return url

    def cancel_ticket(self, request=None):
        if request:
            logger.info(
                "ticket_cancel, user %s has cancelled ticket %s", request.user, self.id
            )
            log_entry(
                type="usage",
                event="test cancelled",
                user=request.user,
                executor=request.user,
                json={"ticket": self.id},
                ip=request.META.get("REMOTE_ADDR"),
            )
        self.cancelled = True
        self.save(update_fields=["cancelled"])
        TicketSignalDispatcher.send_ticket_cancelled(ticket_id=self.pk)

    def custom_force_close(self):
        self.finalize_and_close()
        self.save(update_fields=["close_date"])

    def similarity_check(self):
        if self.similarity_check_enabled:
            from codility.similarity.tasks import ti_analyze_ticket

            logger.info(
                "similarity_check: ticket_close, creating ticket inspection task for %s",
                (self.id),
            )
            self.ti_rpts_ready = False

            # ti_analyze_ticket runs a celery task for similarity that needs to wait for the
            # transaction in case some submits are created in the same view
            transaction.on_commit(lambda: ti_analyze_ticket(self.id))

    def close(self, t=None):
        from codility.codelive.models import MiroCanvas
        from codility.celery_tasks.codelive import (
            make_miro_board_read_only,
        )

        if t is None:
            t = datetime.utcnow()
        self.close_date = t
        self.end = t

        if self.is_codelive:
            miro_board: Optional[MiroCanvas] = MiroCanvas.objects.filter(
                interview=self.codelive_session
            ).first()
            if miro_board is not None and miro_board.miro_board_id is not None:
                make_miro_board_read_only.delay(miro_board.miro_board_id)

        if self.is_codelive and self.codelive_session.candidate_start_date is None:
            self.cancel_ticket()

        if self.submit_set.filter(mode="final", _eval_rpt=None).count() == 0:
            # Set the result if there are no outstanding submits
            # TODO this should be done in one place only,
            # currently it's here and in the checker
            self.result = sum(
                tt.result for tt in self.tickettask_set.all() if tt.result is not None
            )

            # if ticket is assessed it does not have modified score
            # so we result_pr_int should be equal to final_result_pr_int
            self.modified_result_percent = self.result_pr_int

        self.similarity_check()

        # If this is an extended ticket, and its survey was marked as 'read',
        # we want to unmark this, so that we won't miss tickets (Trac #2943)
        from codility.surveys.models import CandidateSurvey

        try:
            survey = CandidateSurvey.objects.get(ticket=self)
        except ObjectDoesNotExist:
            pass
        else:
            if survey.notes and survey.notes.endswith("-"):
                survey.notes += (
                    " (warning: the marker referred to the ticket before extending)"
                )
                survey.save()

        self.reset()
        for tt in self.tickettasks:
            tt.save_effective_time_used()
        self.assign_reviewer()
        TicketSignalDispatcher.send_ticket_closed(self.pk)
        if settings.ASSII_ENABLED:
            AssiiClientBuilder.build_assii_client().post_assessment_event(
                AssessmentEventType.ASSESSMENT_FINISHED, self
            )

    def finalize_and_close(self):
        "Create submits for remaining tasks, and close the ticket"
        assert self.start_date is not None and self.close_date is None

        max_time = self.start or self.start_date
        tickettasks: List[TicketTask] = self.tickettasks
        for ticket_task in tickettasks:
            if not ticket_task.close_date:
                # Check if the task has been opened
                task_start = ticket_task.start_date
                if not task_start:
                    continue
                max_time = max(max_time, task_start)

                # Finished already - just update max time
                task_end = ticket_task.close_date
                if task_end:
                    max_time = max(max_time, task_end)
                    continue

                # Try to find a final solution
                if not ticket_task.solutions:
                    continue

                solution_snapshot: CodeSnapshot = ticket_task.solutions[-1]
                code = solution_snapshot.code
                facade_version = solution_snapshot.facade_version
                solution_tar_gz_version_id = (
                    solution_snapshot.solution_tar_gz_version_id
                )
                # Set solution timestamp to 1 second after the last save,
                # since it's possibly a "verify" request and we don't want
                # the events to overlap.
                timestamp = solution_snapshot.timestamp + timedelta(seconds=1)
                logger.info(
                    "creating auto-submit, ticket: %s, task: %s, task_context: %s",
                    self.id,
                    ticket_task.task.name,
                    solution_snapshot.task_context,
                )
                self.set_task_end(ticket_task.task.name, timestamp)
                self._set_option("auto.%s" % ticket_task.task.name, "1")
                task_context = solution_snapshot.task_context

                solution_path = None
                if facade_version:
                    solution_path = self.get_latest_solution_path(
                        facade_version, task_context, ticket_task
                    )
                evaluation_info_id = None
                if solution_tar_gz_version_id:
                    server_format_task_context = (
                        change_context_from_frontend_to_task_server_format(
                            task_context, ticket_task.task.name
                        )
                    )
                    evaluation_info = EvaluationInfo.objects.only(
                        "solution_tar_gz_object_uri", "pk"
                    ).get(
                        ticket=self.id,
                        task_name=ticket_task.task.name,
                        task_context=server_format_task_context,
                    )
                    solution_path = evaluation_info.solution_tar_gz_object_uri
                    evaluation_info_id = evaluation_info.pk
                self._create_submit(
                    mode="final",
                    task_name=ticket_task.task.name,
                    solution=code,
                    timestamp=timestamp,
                    task_context=task_context,
                    file_submit=self.get_latest_file_submit(ticket_task.task.name),
                    solution_path=solution_path,
                    solution_tar_gz_version_id=solution_tar_gz_version_id,
                    evaluation_info_id=evaluation_info_id,
                )
                max_time = max(max_time, timestamp)
        self.close()
        self.end = max_time
        self.save()

    def get_latest_file_submit(self, task_name: str) -> Optional["FileSubmit"]:
        result: Optional["FileSubmit"] = None
        task_submits = self.task_of_ticket(task_name).submits
        submits_with_file = [
            submit for submit in task_submits if submit.file_submit is not None
        ]
        if submits_with_file:
            result = submits_with_file[-1].file_submit
        else:
            result = None

        # None result may be a symptom of some bugs, especially if happens very often
        # let's log it.
        if result is None:
            logger.info(
                f"Couldn't find latest file submit for ticket: {self.id} and task: {task_name}"
            )

        return result

    @staticmethod
    def get_latest_solution_path(facade_version, task_context, ticket_task):
        from codility.candidate.api.utils import (
            get_facade_repository_by_task_id_and_context,
            get_solution_path,
        )

        latest_interaction: CandidateInteraction = (
            ticket_task.get_newest_candidate_interactions()
        )
        if not latest_interaction:
            logger.warning(
                "No last candidate interaction found for TicketTask=%s", ticket_task.id
            )
        if latest_interaction and latest_interaction.task_context != task_context:
            logger.warning(
                "Task context from latest interaction differs from solution snapshot TicketTask=%s, "
                "task_context: %s, candidate_interaction_task_context: %s",
                ticket_task.id,
                str(task_context),
                str(latest_interaction.task_context),
            )

        facade_repository: FacadeRepository = (
            get_facade_repository_by_task_id_and_context(ticket_task.id, task_context)
        )

        return get_solution_path(facade_repository.bucket_id, facade_version)

    def open(self, t=None, candidate=None):
        assert self.is_valid_to_start, "Ticket have to be valid when opening it"
        if t is None:
            t = datetime.utcnow()
        self.timelimit = self.time_remaining_sec
        self.start_date = t
        self.start = t
        self.filter_unavailable_prg_langs()
        if candidate and candidate.is_authenticated:
            # TODO: Codelive tickets have to be handled separately.
            if not self.is_codelive:
                self.candidate = candidate
        if settings.ASSII_ENABLED:
            AssiiClientBuilder.build_assii_client().post_assessment_event(
                AssessmentEventType.ASSESSMENT_STARTED, self
            )

    @atomic
    def filter_unavailable_prg_langs(self):
        tickettasks: List[TicketTask] = self.tickettasks
        for ticket_task in tickettasks:
            ticket_task.filter_unavailable_prg_langs()
            available_variants = ticket_task.get_available_variants_with_complex_types()
            available_variants.pop(Id("human_lang"), None)
            try:
                ticket_task.filter_unavailable_prg_lang_variants()
            except Exception:
                logger.error(
                    "Error during filtering unavailable variants for ticket %s",
                    self.id,
                    exc_info=True,
                )

            if not available_variants:
                raise Ticket.CouldNotOpen()

    class CouldNotOpen(Exception):
        """
        Thrown if the it's not possible to open a ticket because
        A ticket opened this way would be problematic.
        """

    @property
    def email_recipients(self) -> List[str]:
        return self.rpt_email.splitlines() if self.rpt_email else []

    def send_notifications(self):
        if self.origin == "try":
            return

        from codility.integrations.slack.utils import send_slack_notification
        from codility.integrations.workable.utils import upload_result_to_workable
        from codility.integrations.icims.utils import upload_result_to_icims
        from codility.integrations.smartrecruiters.utils import (
            upload_result_to_smartrecruiters,
        )
        from codility.integrations.smartrecruiters_integration.interface import (
            upload_result_to_smartrecruiters_v2,
        )
        from codility.integrations.sap_successfactors.interface import (
            upload_result_to_sap_successfactors,
        )
        from codility.integrations.jobvite.interface import (
            upload_result_to_jobvite,
        )
        from codility.integrations.eightfold.interface import (
            upload_result_to_eightfold,
        )
        from codility.integrations.linkedin_talenthub.utils import (
            upload_result_to_talenthub,
        )

        all_notification_tasks = [
            # These should all be functions that take a ticket.
            Ticket.send_report,
            send_slack_notification,
            upload_result_to_workable,
            upload_result_to_icims,
            upload_result_to_smartrecruiters,
            upload_result_to_smartrecruiters_v2,
            upload_result_to_sap_successfactors,
            upload_result_to_jobvite,
            upload_result_to_eightfold,
            upload_result_to_talenthub,
        ]
        success = True
        for task in all_notification_tasks:
            try:
                with atomic():
                    assert (
                        len(task.__name__) > 1
                    ), "Each task should have a name for logging"
                    task(self)
            except RequestException:
                success = False
                logger.exception(
                    "send_notifications: could not send a task: %s", task.__name__
                )
                # move on to other tasks
        if success:
            self.notifications_finish_date = datetime.utcnow()
            self.save()

    def reset_notifications(self):
        """
        Remove the records of the notifications being sent, so that they can be sent again.

        This is concerned with e-mail and IM notifications.
        Note: We don't mess with ATS integrations here - they don't support re-opening tickets,
        let's not update them unexpectedly.
        """
        self._remove_option("rpt_email_sent")
        self._remove_option("slack_notification_sent")
        self.notifications_finish_date = None
        self.save()

    def send_report(self):
        # In case we got to a ticket too late.
        if self.rpt_email_sent:
            return

        email_recipients: List[str] = self.email_recipients

        full_email_recipients: List[str] = exclude_report_only_recipients(
            email_recipients
        )
        full_email: RenderedEmailContent = render_email_report(
            self, is_anonymized=False
        )

        logger.info(
            "sending full reports for ticket, %s, # of recipients: %d",
            self.id,
            len(full_email_recipients),
        )

        self._send_email(full_email_recipients, full_email)

        anonymized_email: RenderedEmailContent = render_email_report(
            self, is_anonymized=True
        )
        anonymized_email_recipients: Set[str] = set(email_recipients) - set(
            full_email_recipients
        )

        logger.info(
            "sending anonymized reports for ticket, %s, # of recipients: %d",
            self.id,
            len(anonymized_email_recipients),
        )

        self._send_email(anonymized_email_recipients, anonymized_email)

        logger.info(
            "sent reports for ticket, %s",
            self.id,
        )
        self.rpt_email_sent = 1

    def _send_email(
        self, email_recipients: Iterable[str], email_content: RenderedEmailContent
    ) -> None:
        for to_addr in email_recipients:
            logger.info("sending report, %s, %s", self.id, to_addr)
            send_system_email(
                to_addr=to_addr,
                from_email=settings.NOREPLY_EMAIL,
                subject=email_content.title,
                message=email_content.content,
                html_body=True,
            )

    def should_send_reminder(self):
        if self.creator and self.creator.account.get_options().disable_reminders:
            return False
        return (
            self.email
            and self.creator is not None
            and self.creator.is_recruiter
            and self.start_date is None
            and not self.extended
            and not self.cancelled
            and not self.removed
        )

    def get_absolute_url(self):
        return reverse("ticket_detail", kwargs={"id": self.id})

    # TICKET OPTION LIST

    # simple options
    brand = TicketOptionField("brand")
    start = TicketOptionField("start", timestamp_to_dt, dt_to_timestamp)
    end = TicketOptionField("end", timestamp_to_dt, dt_to_timestamp)
    valid_from = TicketOptionField("valid_from", timestamp_to_dt, dt_to_timestamp)
    valid_to = TicketOptionField("valid_to", timestamp_to_dt, dt_to_timestamp)
    sharp_end_time = TicketOptionField(
        "sharp_end_time", timestamp_to_dt, dt_to_timestamp
    )  # if set timelimit is minimum of normal timelimit and (sharp_end_time - current time)
    rpt_email = TicketOptionField("rpt_email")
    rpt_for_candidate = TicketOptionField("rpt_for_candidate", str_to_int)
    scoring = TicketOptionField(
        "scoring"
    )  # standard, func, correctness, !!! never set !!!
    old_candidate_ip = TicketOptionField("candidate_ip")
    verify = TicketOptionField("verify", timestamp_to_dt, dt_to_timestamp)

    rpt_email_sent = TicketOptionField("rpt_email_sent")
    slack_notification_sent = TicketOptionField("slack_notification_sent")

    prg_lang_list_tmp = TicketOptionField(
        "prg_lang_list", str_to_list, list_to_str
    )  # deprecated, used only by randomizer and historical data

    human_lang_list = TicketOptionField("human_lang_list", str_to_list, list_to_str)
    with_disability = models.BooleanField(null=True, blank=False, default=False)
    cert_test = TicketOptionField("cert_test", str_to_int)
    cert_title = TicketOptionField("cert_title")
    cert_expire = TicketOptionField("cert_expire", timestamp_to_dt, dt_to_timestamp)
    contest_test = TicketOptionField("contest_test", str_to_int)
    exit_url_api = TicketOptionField(
        "exit_url_api"
    )  # plain http redirect after completion
    user_notes = models.TextField(
        default="", null=True, blank=True
    )  # TODO remove null=True?
    which_cert_granted = TicketOptionField("which_cert_granted")
    intro_info_for_cand = TicketOptionField("intro_info_for_cand")
    was_resignation = TicketOptionField("was_resignation")
    verify_all_tests = TicketOptionField(
        "verify_all_tests", str_to_bool, bool_to_str, default_value=False
    )
    demo_task = TicketOptionField("demo_task")
    organization = TicketOptionField("organization")
    trackers = TicketOptionField("trackers", str_to_dict, dict_to_str)
    data_survey_id = TicketOptionField(
        "data_survey_id"
    )  # for training tickets, see programmers.models.ProgrammerDataSurvey
    accessibility_mode = TicketOptionField("accessibility_mode", str_to_int, int_to_str)

    editor = TicketOptionField("editor")  # See candidate_run for available editors
    candidate_ip_list = TicketOptionField(
        "candidate_ip_list", str_to_list, list_to_str
    )  # if test accessed from multiple IPs
    extended = models.IntegerField(null=True, blank=False, default=0)

    ti_rpts_finished = TicketOptionField(
        "ti_rpts_finished", timestamp_to_dt, dt_to_timestamp
    )
    ti_rpts_verified = TicketOptionField(
        "ti_rpts_verified", timestamp_to_dt, dt_to_timestamp
    )
    try_original_ticket_id = TicketOptionField("try_original_ticket_id")
    try_original_solution_id = TicketOptionField("try_original_solution_id")
    codelive_twilio_room_id = TicketOptionField("codelive_twilio_room_id")
    enable_beta_testing = TicketOptionField("enable_beta_testing")

    # end.TASK_NAME
    def set_task_end(self, task_name, dt=None):
        if dt is None:
            dt = datetime.utcnow()
        self._set_option("end." + str(task_name), dt_to_timestamp(dt))

    def get_task_end(self, task_name):
        return timestamp_to_dt(self._get_option("end." + str(task_name)))

    def set_task_start(self, task_name, dt=None, overwrite=True):
        if self._has_option("start." + task_name) and not overwrite:
            return
        if dt is None:
            dt = datetime.utcnow()
        self._set_option("start." + str(task_name), dt_to_timestamp(dt))

    def get_task_start(self, task_name):
        return timestamp_to_dt(self._get_option("start." + str(task_name)))

    # methods

    def save(self, *args, **kwargs):
        logger.debug(
            "ticket save: ticket, %s, %s",
            "%s" % (self.id),
            " ; " + args.__str__() + " ; " + kwargs.__str__(),
        )
        assert self.origin in dict(Ticket.ORIGIN_VALUES), "origin must be valid"
        if self.identity:
            self.identity.last_activity = datetime.now()
            self.identity.save()

        ret = super(Ticket, self).save(*args, **kwargs)
        return ret

    @memo
    def task_changes(ticket):
        """Used to calculate exact time spent on every task in a
        ticket (only looks at last reopen-attempt).  Powers the slider
        in UI."""
        start_time = ticket.start_date
        if ticket.is_codelive:
            if session_started_date := ticket.codelive_session.session_started_date:
                start_time = session_started_date
        if start_time:
            start_time = start_time.replace(microsecond=0)
        events = ticket.events_list
        if not start_time:
            if events:
                start_time = events[0].timestamp
            else:
                start_time = datetime.utcnow()

        task_changes = []
        for event in events:
            if event.timestamp < start_time:
                continue
            task_name = event.task_name
            if event.kind in ("final", "end_task"):
                task_name = None
            task_changes.append({"task_name": task_name, "timestamp": event.timestamp})
        end_time = ticket.close_date or datetime.utcnow()
        return {
            "task_changes": task_changes,
            "start_time": start_time,
            "end_time": end_time,
        }

    def show_countdown_if_not_yet_open(self):
        # TODO enable for contests, etc
        return False

    def get_or_create_share_token(self, days=SHARE_TOKEN_DEFAULT_DURATION):
        end_date = date.today() + timedelta(
            days=days - 1
        )  # subtract 1 since the condition is start <= today <= end

        try:
            token, _ = TicketDetailShareToken.objects.get_or_create(
                ticket=self, start_date=date.today(), end_date=end_date
            )
            return token
        except (
            MultipleObjectsReturned
        ):  # we already have more than one return the latest
            token = TicketDetailShareToken.objects.filter(ticket=self).order_by(
                "-end_date",
                "-id",
            )[0]
            return token

    def find_share_token(self, token_content):
        results = self.ticketdetailsharetoken_set.filter(content=token_content)
        if not results.exists():
            return None
        return results[0]

    # Ticket locking logic, replicated from checker.
    # Intended to be used by ticket maintenance tasks (Celery)
    # that can be repeated later.

    @staticmethod
    def try_lock(id, seconds, now=None, name="checker_lock"):
        """Try locking a ticket for modification.
        Return the lock on success, None on failure."""

        now = now or datetime.utcnow()
        now_ts = dt_to_timestamp(now)
        lock_ts = now_ts + seconds
        logger.info("try_lock, ticket id = %s", id)
        while True:
            try:
                opt, created = TicketOption.objects.get_or_create(
                    ticket_id=id, opt=name, defaults={"data": str(lock_ts)}
                )
            except IntegrityError:
                # Can happen on simultaneous lock attempt.
                # Since we don't retry on failure, just return.
                logger.warning(
                    "try_lock failed with IntegrityError, ticket id = %s", id
                )
                return None

            if created:
                logger.info("try_lock succeeded, ticket id = %s", id)
                return opt
            elif int(opt.data) >= now_ts:
                # There is a still valid lock.
                logger.info("try_lock failed, ticket id = %s", id)
                return None
            else:
                # The lock exists, but is outdated.
                # Delete it and try again.
                opt.delete()
                continue

    @staticmethod
    def unlock(opt):
        try:
            opt.delete()
        except ObjectDoesNotExist:
            logger.warning("trying to unlock a non-existent lock, ticket id = %s", id)

    # Decorator for tasks using locks.
    @staticmethod
    def run_if_unlocked(seconds, lock_name="checker_lock"):
        def decorator(func):
            @wraps(func)
            def wrapped(ticket_id):
                lock = Ticket.try_lock(ticket_id, seconds=seconds)
                if not lock:
                    return
                try:
                    return func(ticket_id)
                finally:
                    Ticket.unlock(lock)

            return wrapped

        return decorator

    @property
    def similarity_check_enabled(self):
        if not self.is_billable:
            return False

        if self.is_codelive:
            return False

        # Okay if the user has similarity check now, or at least had it back when this ticket was created.
        # (no need to say thank you, dear users!)
        user_has_similarity_check = (
            self.creator is not None and self.creator.account.can_use_similarity_check()
        )
        user_had_similarity_check = (
            self.invoice is not None
            and self.invoice.allow_similarity_check in (True, None)
            and not self.invoice.is_free_trial
        )
        return user_has_similarity_check or user_had_similarity_check

    @property
    def similarity_status(self) -> str:
        """Machine-readable similarity status."""
        from codility.similarity.models import TicketInspectionRpt

        # This is a hacky solution to avoid showing "not-found" while the similarity service is still working on the solutions.
        # We assume that the similarity service will finish within 10 minutes after the ticket is closed.
        # The proper solution would require a complete redesign of this unholy disgrace of a status flow.
        # At the this time we think we will have the time and resources for a redesign in the foreseeable future.
        if not self.is_closed or (
            self.close_date and self.close_date + timedelta(minutes=10) > datetime.now()
        ):
            return "not-available"
        elif self.ti_rpts_ready is False:
            return "pending"
        elif self.ti_rpts_ready is True:
            reports = self.get_visible_ti_rpts()
            if any(
                report.rpt_type == TicketInspectionRpt.RPT_SIM_SKIPPED
                for report in reports
            ):
                return "skipped"
            if any(report.is_accepted_by_client is None for report in reports):
                return "found"
            elif any(report.is_accepted_by_client is True for report in reports):
                return "confirmed"
            else:
                return "not-confirmed"
        else:  # ti_rpts_ready is None
            return "not-found"

    @property
    def similarity_status_dict(self) -> Dict[str, typing.Tuple[str, str, str, bool]]:
        return {
            "not-available": (
                "",
                "not available yet",
                "Similarity check will be performed after the test session finishes.",
                False,
            ),
            "pending": (
                "in review",
                "in review",
                "Similarity check is in progress.",
                False,
            ),
            "found": (
                "please resolve",
                "please resolve",
                "Similar solutions have been detected, please resolve.",
                True,
            ),
            "confirmed": (
                "acknowledged",
                "acknowledged",
                "Similarity has been acknowledged.",
                True,
            ),
            "not-confirmed": (
                "dismissed",
                "dismissed",
                "Similarity has been dismissed.",
                True,
            ),
            "not-found": (
                "",
                "not found",
                "No similar solutions have been detected.",
                False,
            ),
            "skipped": (
                "skipped",
                "skipped",
                "The Candidate did not reach the minimum score for Plagiarism check.",
                True,
            ),
        }

    @property
    def similarity_status_description(self) -> Dict[str, Union[str, bool]]:
        status = self.similarity_status

        text_short, text_long, explanation, show_link = self.similarity_status_dict[
            status
        ]

        return {
            "status": status,
            "text_short": text_short,
            "text_long": text_long,
            "explanation": explanation,
            "show_link": show_link,
        }

    def get_visible_ti_rpts(self, admin=False):
        if self.ti_rpts_ready is not True and not admin:
            return []

        reports = self.ticketinspectionrpt_set.filter(is_cancelled=False)
        reports = [rpt for rpt in reports if admin or rpt.is_visible_to_clients]
        return reports

    def is_the_same_candidate(self, other_ticket):
        return self.identity is not None and self.identity == other_ticket.identity

    class Meta:
        db_table = "tickets"
        get_latest_by = "create_date"
        ordering = ["-create_date"]
        base_manager_name = "objects"
        permissions = (
            ("can_create_campaigns", "Can create campaigns"),
            ("can_export_to_csv", "Can export tickets details to CSV format"),
        )

    @property
    def nick_for_email(self):
        """
        One of:
            John Doe (john@example.com)
            John Doe
            john@example.com
            Anonymous
        """
        if self.nick:
            return self.nick + (" (%s)" % self.email if self.email else "")
        else:
            return self.email or "Anonymous"

    @property
    def nick_or_default(self):
        return self.nick or self.default_nick

    @property
    def default_nick(self):
        if self.origin == "demo":
            return "Demo ticket"
        elif self.origin == "training":
            return "Training ticket"
        elif self.origin == "try":
            original_ticket = self.find_original_ticket()
            if original_ticket:
                return f"[Testing] {original_ticket.nick_or_default}"
            elif self.campaign:
                return f"[Testing {self.campaign.name}]"
            elif len(self.tickettasks) == 1:
                task = self.tickettasks[0].task
                return f"[Testing task {task.pretty_name}]"
            else:
                return "[Testing]"
        else:
            return "Anonymous"

    def find_original_ticket(self):
        """Find original ticket, recursively."""

        if not self.try_original_ticket_id:
            return None

        t = self
        seen = set()
        while t.try_original_ticket_id:
            if t.id in seen:
                logger.error(
                    "try_original_ticket_id loop detected for ticket %s", self.id
                )
                return None
            seen.add(t.id)
            try:
                t = Ticket.objects.get(id=t.try_original_ticket_id)
            except Ticket.DoesNotExist:
                logger.error("try_original_ticket_id for ticket %s doesn't exist", t.id)
                return None
        return t

    @property
    def effective_team(self):
        if self.team:
            return self.team
        elif self.campaign:
            return self.campaign.team
        else:
            return None

    def get_account_id_based_on_closest_membership(self):
        if self.creator:
            return self.creator.account_id
        elif self.campaign:
            return self.campaign.account_id
        elif self.team:
            return self.team.account_id
        return None

    @property
    def id_verification_enabled(self):
        return (
            (self.campaign.id_verification_enabled if self.campaign else False)
            and not self.is_codelive
            and (self.is_public or self.is_private)
        )

    @property
    def id_verification_results(self):
        return self.candidateverification_set.first()

    @property
    def is_accessed_from_multiple_networks(self):
        qs: QuerySet = self.access_set.exclude(
            ip=settings.SENTRY_IP_EXCLUDED_FROM_TICKET_ACCESS_DETAILS
        ).distinct("ip")

        return qs.count() > 1

    def get_context(self):
        return {}

    def assign_reviewer(self) -> None:
        from codility.tickets.review_assigners import (
            RandomAssignmentStrategy,
            ReviewerAlreadyAssignedError,
            ReviewAssigner,
        )

        if any((self.is_codelive, self.is_training, self.is_try)):
            return
        assigner = ReviewAssigner(assignment_strategy=RandomAssignmentStrategy)
        try:
            assigner.assign_closing_ticket(ticket=self)
        except ReviewerAlreadyAssignedError:
            pass

    def reopen(self, time_limit: int, task_names: List[str]) -> None:
        self.start_date = None
        self.close_date = None
        self.result = None
        self.candidate = None
        # HACK: Extended tickets shouldn't be hidden from the user again.
        if self.is_public:
            self.origin = "private"
        self._set_former_time()
        self.timelimit = time_limit * 60
        self._remove_option("start")
        self._remove_option("end")
        all_task_names = self.get_task_names()
        for task_name in all_task_names:
            t_start = self.get_task_start(task_name)
            if t_start:
                if task_name in task_names:
                    self._remove_option("end.%s" % task_name)
                elif not self.get_task_end(task_name):
                    self._set_option("end.%s" % task_name, dt_to_timestamp(t_start))
        self._remove_option("prg_lang_list")
        self.human_lang_list = ["en"]
        for tt in self.tickettasks:
            prg_lang_variant = (
                {"prg_lang": tt.prg_lang_list} if tt.prg_lang_list else {}
            )

            # TODO: The _variants... flow is currently experimental.
            #  Remove try-except clause once variants are fully integrated to the system.
            try:
                TicketTaskVariantHandler.set_variants_available_at_ticket_reopen(
                    tickettask=tt,
                    variants={**tt.get_variants_available(), **prg_lang_variant},
                )
            except Exception:
                logger.exception(
                    "Error in experimental variants flow while saving changes to ticket"
                )
        self.save()

        self.notify_reopened()

    def notify_reopened(self) -> None:
        days_before_invites_expire: Optional[int] = (
            self.campaign.days_before_invites_expire if self.campaign else None
        )

        TicketSignalDispatcher.send_ticket_reopened(
            ticket_id=self.pk,
            days_before_invites_expire=days_before_invites_expire,
        )

    def _set_former_time(self) -> None:
        start = int(self._get_option("start", 0))
        end = int(self._get_option("end", 0))
        if start and end and start != end:
            former = int(self._get_option("former_time", 0))
            self._set_option("former_time", str(former + end - start))
        if start or self.extended:
            self.extended = 1

    def __str__(self):
        if self.nick or self.email:
            return "%s: %s" % (self.id, self.nick or self.email)
        else:
            return "%s" % self.id


def get_reply_to(fallback_email, creator=None, events=()):
    filtered_send_events = [e for e in events if e.kind == "send"]
    filtered_reply_to_events = [
        e for e in filtered_send_events if e.data.get("headers", {}).get("Reply-To", "")
    ]
    sorted_events = sorted(
        filtered_reply_to_events, key=lambda e: e.timestamp, reverse=True
    )

    if len(sorted_events) > 0:
        return sorted_events[0].data.get("headers", {}).get("Reply-To")
    elif creator and creator.is_recruiter and creator.email:
        return creator.email
    else:
        return fallback_email


class ModifiedResult(models.Model):
    RESULT_TYPES = Choices(("evaluated", "Evaluated"), ("custom", "Custom"))
    tickettask = models.ForeignKey("TicketTask", on_delete=models.CASCADE)
    result = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    modifier = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    modifier_notes = models.TextField(default="", null=True, blank=True)
    create_date = models.DateTimeField(auto_now_add=True)
    result_type = models.CharField(
        max_length=16, choices=RESULT_TYPES, default=RESULT_TYPES.evaluated
    )

    @classmethod
    def generate_modified_result_description(
        cls,
        result_type,
        result=None,
        original_result=None,
        modifier_email=None,
        set_date=None,
    ):
        final_result = result if result is not None else original_result
        result_text = (
            str(int(final_result)) + "%" if final_result is not None else "N/A"
        )
        description = "" + result_text
        if result_type == cls.RESULT_TYPES.evaluated:
            description += " (Codility result)"
        if modifier_email is not None:
            description += " " + modifier_email
        if set_date is not None:
            description += str(set_date.strftime(" %b %d, %Y %H:%M"))
        return description

    def description(self):
        return self.generate_modified_result_description(
            self.result_type,
            self.result,
            self.tickettask.result,
            self.modifier.email,
            self.create_date,
        )


class FileSubmit(models.Model):
    file_key = models.CharField(max_length=256, null=True, blank=True)
    tickettask = models.ForeignKey("TicketTask", on_delete=models.CASCADE)
    permission_request_date = models.DateTimeField(auto_now_add=True)

    def get_absolute_url(self):
        return reverse(
            "ticket_task_file",
            kwargs={
                "id": self.tickettask.ticket_id,
                "task_name": self.tickettask.task.name,
            },
        )


class SubmitQuerySet(models.QuerySet):
    def with_tasks(self) -> List["Submit"]:
        """Return submits with prefetched task objects."""
        submits = list(self)
        task_names = {s.task for s in submits}
        task_name_to_obj = {t.name: t for t in Task.objects.filter(name__in=task_names)}
        assert set(task_name_to_obj) == task_names

        for submit in submits:
            submit._task_object = task_name_to_obj[submit.task]

        return submits


class SubmitManager(models.Manager):
    def create(
        self,
        task: str,
        prg_lang: Optional[str] = None,
        _task_context: Optional[dict] = None,
        **kwargs,
    ) -> "Submit":
        assert _task_context is not None
        assert prg_lang is None

        try:
            get_task_info_from_task_name(task)
        except TasksNotFoundError as e:
            raise ValueError(f"Unknown task: {task}") from e

        return super().create(
            task=task,
            prg_lang=_task_context.get("prg_lang") or "",
            _task_context=_task_context,
            **kwargs,
        )

    def all_unhandled(self, mode=None):
        qs = self.filter(eval_date__isnull=True)
        if mode is not None:
            qs = qs.filter(mode=mode)
        return qs

    def store_checker_results(self, id, results):
        # id deliberately ignored, normal checker sends it in results too
        from .checker import store_checker_results

        store_checker_results(results)

    def get_queryset(self) -> SubmitQuerySet:
        return SubmitQuerySet(self.model, using=self._db)


def clean_submit_task_context(obj: "Submit", context: Dict[str, str]) -> Dict[str, str]:
    return clean_task_context_for_task_info(obj.task_info, task_context=context)


def similarity_check_enabled(task_context: Dict[str, str], solution: str, mode: str):
    prg_lang = task_context.get("prg_lang")
    if not prg_lang:
        return False

    MIN_SOL_LENGTH = 50
    if prg_lang in ["cs", "java", "scala"]:
        MIN_SOL_LINES = 12
    else:
        MIN_SOL_LINES = 8

    def valid_line(line):
        if re.match(r"^\s*$", line):
            return False
        if re.match(r"^\s*//.*$", line):
            return False
        if re.match(r"^import.*", line):
            return False
        if re.match(r"^\s*#include.*", line):
            return False
        return True

    from codility.spra_similarity.parsing import (
        parse_source,
        rebuild_parsed_source,
        ParseError,
    )

    source = solution

    try:
        source = rebuild_parsed_source(source, parse_source(source, prg_lang))
    except ParseError:
        logger.warning(f"Parse error for {prg_lang}, Continue without uniformization.")
    except ClassNotFound:
        logger.warning(
            f"Impossible to parse solution for {prg_lang}. Disabling similarity check."
        )
        return False
    except AttributeError:
        logger.warning("Similarity check for Own tasks are temporarily disabled.")
        return False

    lines = len([x for x in source.splitlines() if valid_line(x)])

    if mode == "final" and len(source) >= MIN_SOL_LENGTH and lines >= MIN_SOL_LINES:
        return True
    else:
        return False


class Submit(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE, db_column="ticket")
    task = models.CharField(max_length=64)
    mode = models.CharField(max_length=32)
    _task_context = jsonfield.JSONField(
        null=True,
        blank=False,
        default=dict,
        db_column="task_context",
        help_text="Dictionary of task variants (e.g. prg_lang, human_lang)",
    )
    task_context = TaskContextGetter(cleaner_func=clean_submit_task_context)
    prg_lang = models.CharField(
        max_length=32
    )  # Deprecated -- use `task_context['prg_lang']`

    # Date when the solution was submitted. If the submission is automatic,
    # submit_date is real, while the date of associated CodeSnapshot reflects
    # the time when candidate finished solving the task.
    submit_date = models.DateTimeField()
    # Date when the assessment was started.
    eval_start_date = models.DateTimeField(null=True, blank=True)
    # Date when the assessment was finished.
    eval_date = models.DateTimeField(null=True, blank=True)

    # Checker identifier (hostname, e.g. 'checker-bombur').
    evaluated_by = models.CharField(max_length=42, null=True, blank=True)

    verify_date = models.DateTimeField(null=True, blank=True)  # not used, to be removed
    solution = models.TextField(null=True, blank=True)
    # For a url facade+http://localhost/bucket/x?version=y, this should be /bucket/x?version=y.
    solution_path = models.URLField(null=True, blank=True, db_column="solution_url")
    solution_fingerprint = models.TextField(null=True, blank=True)
    file_submit = models.ForeignKey(
        FileSubmit, on_delete=models.CASCADE, null=True, blank=True
    )
    options = models.TextField(null=True, blank=True)
    result = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    _eval_rpt = models.TextField(null=True, blank=True, db_column="eval_rpt")
    badge = models.CharField(max_length=15, null=True, blank=True)
    execution_id = models.CharField(max_length=64, null=True, blank=True)
    solution_tar_gz_version_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="S3 object version",
    )
    # Non-relation field pointing at EvaluationInfo table as a one2many relation - but without producing expensive SQL changes.
    evaluation_info_id = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "submits"
        ordering = ["-submit_date"]

    objects = SubmitManager()

    def __str__(self):
        return "%s: (%s, %s)" % (self.id, self.ticket, self.task)

    def refresh_from_db(self, using=None, fields=None):
        """Overriding the refresh_from_db method to enforce cache-recalculation on eval_rpt."""
        try:
            del self.eval_rpt
        # If the property wasn't accessed before the cache is empty and it will throw an AttributeError.
        except AttributeError:
            pass
        return super().refresh_from_db(using=using, fields=fields)

    def save(self, *args, **kwargs):
        """Overriding the save method to enforce cache-recalculation on eval_rpt."""
        try:
            del self.eval_rpt
        # If the property wasn't accessed before the cache is empty and it will throw an AttributeError.
        except AttributeError:
            pass
        return super().save(*args, **kwargs)

    @cached_property
    def eval_rpt(self) -> str:
        def is_report_from_evaluations_api_enabled(ticket: Ticket) -> bool:
            if account := ticket.get_creator_account():
                group: AccountGroup = AccountGroup.objects.get(
                    name=SUBMIT_REPORT_FROM_EVALUATIONS_API_ENABLED_AG
                )
                return (
                    group is not None and group.accounts.filter(id=account.id).exists()
                )
            return False

        # If the submit is not provided by evaluations API or FF is disabled read from database
        if (
            self.evaluation_info_id is None
            or self.execution_id is None
            or not is_report_from_evaluations_api_enabled(self.ticket)
        ):
            return self._eval_rpt

        # Otherwise try to read the report from Evaluations API
        evaluation_id = EvaluationInfo.objects.get(
            id=self.evaluation_info_id
        ).evaluation_id
        client = EvaluationsApiClientHttp(
            base_url=settings.EVALUATIONS_API_BASE_URL,
            api_key=settings.EVALUATIONS_API_API_KEY,
        )
        endpoint = (
            client.get_verification_report
            if self.mode == "verify"
            else client.get_final_evaluation_report
        )
        try:
            return endpoint(evaluation_id=evaluation_id, execution_id=self.execution_id)
        except EvaluationApiClientException:
            # Ignoring the error here as it is already logged on client layer - fallback silently to reading the database.
            return self._eval_rpt

    def set_eval_rpt(self, report) -> None:
        """
        Allows writing to submit._eval_rpt to provide consistent experience once the eval_rpt property is introduced.
        Doesn't save the object - just like field assignment wouldn't - only invalidates the cache on eval_rpt property.
        """
        try:
            del self.eval_rpt
        # If the property wasn't accessed before the cache is empty and it will throw an AttributeError.
        except AttributeError:
            pass
        self._eval_rpt = report

    @property
    def task_info(self) -> TaskInfo:
        return get_task_info_from_task_name(self.task)

    @property
    def waiting_time(self):
        assert self.eval_date
        return self.eval_date - self.submit_date

    def date_authored(self):
        """when the submit has been coded and sent, to the best of our knowledge"""
        snapshots = self.codesnapshot_set.all()
        if not snapshots:
            return self.submit_date
        return max(sn.timestamp for sn in snapshots)

    @property
    def user_tests(self):
        """returns user tests assigned to this submission"""
        e = self.eval_dict
        if not e or "tests" not in e:
            return None
        user_tests = []
        for test in e["tests"]["test_list"]:
            if test["type"] == "input" and "input" in test:
                user_tests.append(test["input"])
        return user_tests

    @property
    def performance_tests(self):
        if not hasattr(self, "_performance_tests"):
            tmp = self.eval_dict
            if tmp:
                self._performance_tests = json.dumps(tmp["tests"]["groups"]["perf"])
            else:
                self._performance_tests = None
        return self._performance_tests

    @property
    def random_tests(self):
        """returns random tests assigned to this submission"""
        if not hasattr(self, "_random_tests"):
            tmp = self.eval_dict
            if (
                tmp is None
                or "random_tests" not in tmp
                or "test_list" not in tmp["random_tests"]
            ):
                self._random_tests = None
            else:
                res = tmp["random_tests"]["test_list"]
                self._random_tests = res
        return self._random_tests

    @property
    def eval_dict(self):
        if self.eval_rpt:
            try:
                return parse_xml_rpt(self.eval_rpt)
            except Exception:
                logger.exception(
                    "error parsing report, ticket=%s, submit=%s",
                    self.ticket.id,
                    self.id,
                )
                return None
        else:
            return None

    @property
    def test_groups(self):
        return get_test_groups(self.eval_dict, self.mode)

    @property
    def summary(self):
        return get_summary(self.eval_dict, self.mode)

    @property
    def cert_badge(self):
        if not hasattr(self, "_cert_badge"):
            badge = None
            d = self.eval_dict
            if not (d is None):
                badge = d.get("badge", None)
            self._cert_badge = badge
        return self._cert_badge

    @property
    def get_solution_fingerprint(self):
        if self.mode != "final":
            return None

        if self.solution_fingerprint is None:
            from codility.utils.solution_fingerprint import solution_fingerprint

            self.solution_fingerprint = solution_fingerprint(
                self.solution, self.task_context.get("prg_lang") or ""
            )
            self.save()
        return self.solution_fingerprint

    @property
    def verification_result(self):
        if self.mode != "verify":
            return None
        e = self.eval_dict
        if e is None:
            return None
        if "compile" not in e or "result" not in e["compile"]:
            return None
        if e["compile"]["result"] != "1":
            return "Compile error"
        for t in e["tests"]["test_list"]:
            if t["result"] != "OK":
                return t["result"]
        return "OK"

    @property
    def prg_lang_name(self):
        prg_lang = self.task_context.get("prg_lang")
        if prg_lang in PRG_LANGS:
            return PRG_LANGS[prg_lang].name

        return ""

    @property
    def similarity_check_enabled(self):
        return similarity_check_enabled(
            task_context=self.task_context, solution=self.solution, mode=self.mode
        )

    def uploaded_file_key(self):
        if not self.eval_rpt:
            return None
        report = self.eval_dict
        if "file_upload" not in report:
            return None
        return report["file_upload"].get("file_key")

    def uploaded_file_size(self):
        if not self.eval_rpt:
            return None
        report = self.eval_dict
        if "file_upload" not in report:
            return None
        return report["file_upload"].get("file_size")

    def uploaded_file_errors(self):
        if not self.eval_rpt:
            return None
        report = self.eval_dict
        if "file_upload" not in report:
            return None
        return report["file_upload"].get("errors")

    @property
    def tickettask(self):
        return self.ticket.task_of_ticket(self.task)

    @property
    def is_docker_checker_submit(self):
        try:
            task = Task.objects.get(name=self.task)
            return task.is_docker_checker
        except Task.DoesNotExist:
            return False

    def as_checker_request(self):
        from .checker import get_checker_request

        return get_checker_request(self)

    def build_context(self):
        return {
            **self.task_context,
            # Add implicit context from the ticket, and properly overwrite any
            # unexpected changes to the context that should not be modified by
            # the candidate.
            **self.ticket.get_context(),
        }

    def send_to_checker(self, api: Optional[CheckerApi] = None) -> None:
        if self.is_docker_checker_submit:
            kind = "docker"
        else:
            kind = self.mode

        api = api or checker_api(self.ticket.id)
        api.check_submit(kind, **self.as_checker_request())

    def get_partial_reports(self):
        return checker_api(self.ticket.id).get_partial_reports(self.mode, self.id)

    # Used for testing.
    def add_partial_report(self, report, ttl):
        checker_api(self.ticket.id).add_partial_report(self.mode, self.id, report, ttl)

    def request_final_and_store_execution_id_from_evaluations_api(self) -> None:
        """
        Assigns execution_id to the submit by requesting final from Evaluations API.
        """
        if self.mode != "final":
            raise ValueError("Execution ID can be assigned only for final submits")

        evaluation_info = EvaluationInfo.objects.only("pk").get(
            pk=self.evaluation_info_id
        )

        evaluations_api_client = EvaluationsApiClientHttp(
            base_url=settings.EVALUATIONS_API_BASE_URL,
            api_key=settings.EVALUATIONS_API_API_KEY,
        )
        execution_id = evaluations_api_client.request_final(
            evaluation_id=evaluation_info.evaluation_id,
            solution_version=self.solution_tar_gz_version_id,
        ).id

        self.execution_id = execution_id
        self.save()

    def request_verification_and_store_execution_id_from_evaluations_api(self) -> None:
        """
        Assigns execution_id to the submit by requesting verification from Evaluations API.
        """
        if self.mode != "verify":
            raise ValueError("Execution ID can be assigned only for verify submits")

        evaluation_info = EvaluationInfo.objects.only("pk").get(
            pk=self.evaluation_info_id
        )

        evaluations_api_client = EvaluationsApiClientHttp(
            base_url=settings.EVALUATIONS_API_BASE_URL,
            api_key=settings.EVALUATIONS_API_API_KEY,
        )
        execution_id = evaluations_api_client.request_verification(
            evaluation_id=evaluation_info.evaluation_id,
            solution_version=self.solution_tar_gz_version_id,
        ).id

        self.execution_id = execution_id
        self.save()


def submit_post_save(instance: Submit, created: bool, raw: bool, **kwargs) -> None:
    if settings.ENQUEUE_NEW_SUBMITS and created and not raw:
        if not instance.evaluation_info_id:
            # If evaluation_info_id is not set, the submit triggers checkers
            # ommiting the Evaluations API. This is the case for old submits
            method_to_call = instance.send_to_checker
        elif instance.mode == "verify":
            method_to_call = (
                instance.request_verification_and_store_execution_id_from_evaluations_api
            )
        elif instance.mode == "final":
            method_to_call = (
                instance.request_final_and_store_execution_id_from_evaluations_api
            )

        if settings.TESTING:
            method_to_call()
        else:
            transaction.on_commit(lambda: method_to_call())


signals.post_save.connect(submit_post_save, sender=Submit)


class CodeSnapshotManager(models.Manager):
    def get_queryset(self):
        return (
            super(CodeSnapshotManager, self)
            .get_queryset()
            .select_related("task", "submit")
        )

    def create(self, *, _task_context, prg_lang=None, **kwargs) -> "CodeSnapshot":
        assert prg_lang is None
        assert _task_context is not None
        return super().create(
            prg_lang=_task_context.get("prg_lang") or "",
            _task_context=_task_context,
            **kwargs,
        )

    def get_for_version_and_task_info(
        self, task_info: TaskInfo, ticket: Ticket, version: int
    ) -> "CodeSnapshot":
        return self.get(task__name=task_info.name, ticket=ticket, id=version)

    def get_for_version(
        self, task: Task, ticket: Ticket, version: int
    ) -> "CodeSnapshot":
        return self.get(task=task, ticket=ticket, id=version)

    def get_for_facade_version(
        self, task_name: str, ticket: Ticket, facade_version: str
    ) -> "CodeSnapshot":
        return self.get(
            task__name=task_name, ticket=ticket, facade_version=facade_version
        )


def clean_code_snapshot_task_context(
    obj: "CodeSnapshot", context: Dict[str, str]
) -> Dict[str, str]:
    return obj.task.clean_task_context(context)


class CodeSnapshot(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    task: Task = models.ForeignKey(Task, on_delete=models.CASCADE)
    code = models.TextField(null=False, blank=False)
    timestamp: datetime = models.DateTimeField(
        null=False,
        blank=False,
        help_text="Date when the solution was created, to the best of our knowledge.",
    )
    _task_context = jsonfield.JSONField(
        null=True,
        blank=False,
        default=dict,
        db_column="task_context",
        help_text="Dictionary of task variants (e.g. prg_lang, human_lang)",
    )
    task_context = TaskContextGetter(cleaner_func=clean_code_snapshot_task_context)
    prg_lang = models.CharField(
        max_length=32
    )  # Deprecated -- use `task_context['prg_lang']`

    submit: Optional[Submit] = models.ForeignKey(
        Submit, on_delete=models.CASCADE, null=True, blank=True
    )
    facade_version = models.CharField(max_length=255, null=True, blank=True)
    # solution_tar_gz_version_id stores Evaluations API S3 solution version
    solution_tar_gz_version_id = models.CharField(max_length=255, null=True, blank=True)

    objects = CodeSnapshotManager()

    def __init__(self, *args, **kwargs):
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.utcnow()
        if "code" in kwargs and kwargs["code"] is None:
            kwargs["code"] = ""
        super(CodeSnapshot, self).__init__(*args, **kwargs)

    @property
    def mcq_prettyprint(self):
        assert self.task_context.get("prg_lang") == "mcq"

        def answer_to_char(s):
            if isinstance(s, int):
                return string.ascii_uppercase[s]
            elif isinstance(s, list):
                return (
                    "-" if not s else ",".join([string.ascii_uppercase[i] for i in s])
                )
            else:
                return "-"

        try:
            json_answers = json.loads(self.code)["answers"]
            answers = " ".join(
                "%d%s" % (i, answer_to_char(x))
                for i, x in enumerate(json_answers, start=1)
            )
            return "Answers given: %s" % answers
        except (ValueError, IndexError):
            return ""

    @property
    def diff_with_template(self):
        if not self.task.is_bugfixing:
            raise NotImplementedError

        # Strip trailing whitespace, since it's a common modification and it's
        # actually filtered out by CUI and spratools. See Trac #3160.
        additional_data = self.ticket_task.additional_data
        solution_template = self.task.get_solution_template(
            task_context=self.task.clean_task_context(task_context=self.task_context),
            additional_data=additional_data,
        )
        if solution_template is None:
            logger.error(
                "No solution template found for %s, %s, %s",
                self.task.name,
                self.task_context.get("prg_lang") or "(None)",
                additional_data,
            )
            return []
        original_lines = [line.rstrip() for line in solution_template.splitlines()]
        solution_lines = [line.rstrip() for line in self.code.splitlines()]

        n = len(original_lines)  # len(original_lines) should equal len(solution_lines)
        return list(
            difflib.unified_diff(original_lines, solution_lines, "a", "b", "", "", n)
        )

    @property
    def ticket_task(self) -> Optional["TicketTask"]:
        return self.ticket.task_of_ticket(self.task.name)

    @property
    def has_pasted_lines(self):
        res = len(self.get_pasted_lines) > 0
        return res

    @property
    def get_pasted_lines(self):
        res = []
        for p in self.ticket.pasted_codes(self):
            res += list(range(p.start_line, p.end_line + 1))
        return res

    @property
    def is_file_solution(self):
        # note we can't use self.submit.file_submit is None, since this
        # won't work on autosaves
        return self.task_context.get("prg_lang") == "file"


class PastedCode(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    pasted = models.TextField(null=False, blank=False)
    start_pos = models.IntegerField(null=False, blank=False)
    end_pos = models.IntegerField(null=False, blank=False)
    start_line = models.IntegerField(null=False, blank=False)
    end_line = models.IntegerField(null=False, blank=False)
    result_snapshot = models.ForeignKey(
        CodeSnapshot, on_delete=models.CASCADE, null=True, blank=True
    )


class TicketAccess(models.Model):
    ticket = models.ForeignKey(
        Ticket, on_delete=models.CASCADE, related_name="access_set"
    )
    ip = models.GenericIPAddressField()
    http_user_agent = models.TextField(null=False, blank=True, default="")
    http_accept = models.TextField(null=False, blank=True, default="")
    http_accept_encoding = models.TextField(null=False, blank=True, default="")
    http_accept_language = models.TextField(null=False, blank=True, default="")
    geoip_country_code = models.TextField(null=False, blank=True, default="")

    class Meta:
        ordering = ["ip"]
        verbose_name_plural = "ticket accesses"
        unique_together = [
            (
                "ticket",
                "ip",
                "http_user_agent",
                "http_accept",
                "http_accept_encoding",
                "http_accept_language",
                "geoip_country_code",
            )
        ]


@dataclass(frozen=True)
class TicketOptions:
    """The aggregate of all Ticket options available. The TicketOption model is the persistence layer."""

    exit_url_api: Optional[str]
    human_lang_list: Optional[List[str]]
    intro_info_for_cand: Optional[str]
    rpt_email: Optional[str]
    sharp_end_time: Optional[datetime]
    valid_from: Optional[datetime]
    valid_to: Optional[datetime]
    verify_all_tests: bool


class TicketOption(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE, related_name="options")
    opt = models.CharField(max_length=64)
    data = models.TextField(null=True, blank=True)

    def __str__(self):
        return "%s: (%s, %s)" % (self.ticket_id, self.opt, self.data)

    class Meta:
        unique_together = [("ticket", "opt")]
        db_table = "tickets_option"


class TicketTaskManager(models.Manager):
    def get_queryset(self):
        return super(TicketTaskManager, self).get_queryset().select_related("task")

    def create(
        self, task_name: Optional[str] = None, **kwargs: Dict[str, object]
    ) -> "TicketTask":
        uses_task_xor_task_name = (task_name is not None) ^ ("task" in kwargs)
        assert uses_task_xor_task_name, "Cannot use both task_name and task"

        if task_name is not None:
            kwargs["task"] = Task.objects.get(name=task_name)

        return super().create(**kwargs)

    def bulk_create_from_kwargs(
        self, kwargs_list: List[Dict[str, object]]
    ) -> "List[TicketTask]":
        kwargs_with_task_name: Dict[str, Dict[str, object]] = {}

        for task_kwargs in kwargs_list:
            uses_task_xor_task_name = ("task_name" in task_kwargs) ^ (
                "task" in task_kwargs
            )
            assert uses_task_xor_task_name, "Cannot use both task_name and task"

            if "task_name" in task_kwargs:
                kwargs_with_task_name[task_kwargs["task_name"]] = task_kwargs

        expected_task_names = set(kwargs_with_task_name.keys())
        tasks = Task.objects.filter(name__in=expected_task_names)
        filtered_task_names = set(task.name for task in tasks)

        assert (
            filtered_task_names == expected_task_names
        ), f"Some tasks are missing: {expected_task_names - filtered_task_names}"

        for task in tasks:
            kwargs_with_task_name[task.name]["task"] = task
            del kwargs_with_task_name[task.name]["task_name"]

        return self.bulk_create(TicketTask(**kwargs) for kwargs in kwargs_list)


def validate_task_context_for_ticket_task(
    task_context: Dict, tickettask: "TicketTask"
) -> None:
    clean_context = tickettask.clean_task_context(task_context=task_context)
    if clean_context != task_context:
        raise ValidationError(f"Invalid task context: {task_context}")


def validate_custom_task_description(custom_task_description):
    if custom_task_description is not None:
        if "type" not in custom_task_description:
            raise ValidationError('"type" key expected.')
        if (
            custom_task_description["type"]
            not in TicketTask.VALID_TASK_DESCRIPTION_TYPES
        ):
            raise ValidationError("Invalid task description type.")
        if "content" not in custom_task_description:
            raise ValidationError('"content" key expected.')
        if not isinstance(custom_task_description["content"], str):
            raise ValidationError("Content should be a string.")


def validate_prg_lang_list(
    lang_list: List[str], ticket: Optional[Ticket] = None
) -> None:
    if type(lang_list) != list:
        raise ValidationError('This must be a list, e.g. ["c", "cpp"].')
    ticket_variants = []
    if ticket:
        ticket_variants = ticket.all_prg_langs_exts()

    valid_langs = set(list(PRG_LANGS.keys()) + ticket_variants)
    for lang in lang_list:
        if lang not in valid_langs:
            raise ValidationError(
                "%s is not a valid language extension" % lang,
                code="GENERIC_ERROR_WRONG_VALUE",
            )


def validate_prg_lang(lang):
    if lang not in PRG_LANGS:
        raise ValidationError(
            "%s is not a valid language extension" % lang,
            code="GENERIC_ERROR_WRONG_VALUE",
        )


def validate_additional_data(additional_data):
    # TODO: possibily to removal
    if additional_data is not None:
        if not isinstance(additional_data, dict):
            raise ValidationError("This should be a dictionary.")
        if not (
            set(additional_data.keys()) <= {"variant_seed", "variant_specification"}
        ):
            raise ValidationError(
                'The only possible keys are "variant_seed" and "variant_specification".'
            )
        if "variant_seed" in additional_data:
            if not isinstance(additional_data["variant_seed"], str):
                raise ValidationError("Variant seed should be a string.")
        if "variant_specification" in additional_data:
            if not isinstance(additional_data["variant_specification"], dict):
                raise ValidationError("Variant specification should be a dictionary.")


class EvaluationInfo(models.Model):
    evaluation_id = models.UUIDField(null=False)
    ticket = models.ForeignKey(
        Ticket, on_delete=models.CASCADE, related_name="evaluation_info"
    )
    task_name = models.CharField(max_length=300, null=False)
    task_context = models.JSONField(null=True)
    solution_tar_gz_presigned_url = models.CharField(
        max_length=2048,
        null=False,
        help_text=(
            "S3 presigned URL to download tar.gz solution file. For example: "
            "https://solution-evaluation-bucket.s3.amazonaws.com/83707a2b-e8c3-4b23-9695-2afce9f07809.tar.gz?..."
        ),
    )
    solution_tar_gz_object_uri = models.CharField(
        max_length=2048,
        null=False,
        help_text=(
            "S3 object URI to download tar.gz solution file "
            "s3://evaluations-solutions/6f3a20bb-05e6-4e9b-bf5f-97d54f388d65.tar_gz"
        ),
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["ticket", "task_name", "task_context"],
                name="evaluation_info_ticket_task_name_context_key",
            )
        ]


class TicketTask(models.Model):
    ticket: Ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    task: Task = models.ForeignKey(Task, on_delete=models.CASCADE)
    num = models.IntegerField()
    max_result = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )
    result = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    correctness = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )
    performance = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )
    status = models.IntegerField(null=True, blank=True)
    # Stores list of selected languages. Either at the creation time
    # or when the ticket was opened.
    # When opening a ticket, we need to filter unavailable languages out
    # from this list because the availability might have changed between
    # creating the ticket and opening it.
    # If None, all languages are available.
    prg_lang_list = jsonfield.JSONField(null=True, blank=True, default=None)
    latest_access_date = models.DateTimeField(null=True, blank=True)

    additional_data: dict = (
        jsonfield.JSONField(  # TODO: to be removed after agreeing on that
            null=True, blank=True
        )
    )
    task_weight_id = models.CharField(max_length=255, null=True, blank=True)

    VALID_TASK_DESCRIPTION_TYPES = ["text", "html", "txt"]
    DEFAULT_CUSTOM_TASK_DESCRIPTION = {"type": "text", "content": ""}

    # { type: "text", content: "..." }
    custom_task_description: dict = jsonfield.JSONField(
        null=True,
        blank=True,
        default=None,
        validators=[validate_custom_task_description],
    )

    # Effective time information, updated on ticket close.

    _effective_time_used = models.IntegerField(
        null=True, blank=True, db_column="effective_time_used"
    )
    # TODO move to non-null
    _effective_time_used_is_reliable = models.BooleanField(
        null=True, blank=True, db_column="effective_time_used_is_reliable"
    )
    _variants_available_at_create = jsonfield.JSONField(null=True)
    _variants_available_at_open = jsonfield.JSONField(null=True)

    objects = TicketTaskManager()

    def is_description_editable_for(self, user: User, session: SessionBase) -> bool:
        return self.task.has_whiteboard and is_codelive_interviewer(
            self.ticket, user, session
        )

    def clean(self):
        validate_prg_lang_list(self.prg_lang_list, self.ticket)
        super().clean()

    @property
    def _prg_langs(self):
        # TECHDEBT: This is a big hack here. Consider reworking how programming
        #           languages are assigned to a ticket. Especially, for programmers' home
        #           it is safe to assign a new language we want to test run before
        #           showing it to our customers.
        #           Currently, we always scope by languages available to the creator :-(
        # Override the available programming languages for training tickets.
        if self.ticket.origin == "training":
            prg_langs = (
                set(self.task.get_available_programming_languages())
                & prg_lang_visible_in_programmers_home()
            )
        else:
            if self.prg_lang_list:
                prg_langs = set(self.prg_lang_list)
            else:
                prg_langs = set(self.task.get_available_programming_languages())

        # prg_lang_list allows to limit available languages; it does not include
        # special languages.
        if self.ticket.prg_lang_list_tmp:
            allowed_langs = self.ticket.prg_lang_list_tmp + [
                "sql",
                "sql-postgres",
                "file",
                "txt",
                "mcq",
            ]
            prg_langs &= set(allowed_langs)

        return sorted(prg_langs)

    def get_available_variants_with_complex_types(self) -> Dict[Id, Variant]:
        prg_lang_id = Id(get_programming_language_variant_name())
        task_variants = self.task.get_available_variants()
        if prg_lang_id in task_variants:
            task_prg_langs = {
                str(variant_key): variant_value.to_dict()
                for variant_key, variant_value in task_variants[
                    prg_lang_id
                ].values.items()
            }
            # note: if a ticket is not open, self._prg_langs can potentially still contain invalid languages
            # (they will get filtered out during open())
            variant_values = [
                task_prg_langs[prg_lang]
                for prg_lang in self._prg_langs
                if prg_lang in task_prg_langs
            ]
            # If the variant has no valid values, remove it from the dictionary.
            if variant_values:
                task_variants[prg_lang_id] = Variant.build(
                    name=get_programming_language_variant_name(),
                    source={
                        "title": "Programming Language",
                        "select": "one",
                        "values": variant_values,
                    },
                )

            else:
                task_variants.pop(prg_lang_id)

        return task_variants

    def filter_unavailable_prg_langs(self):
        """
        Before a ticket is open, it's allowable to have unsupported languages in self.prg_lang_list.
        This method narrows down self.prg_lang_list to only have supported languages
        (when starting a ticket).
        """
        from codility.languages.models import DeprecatedContext

        self.prg_lang_list = sorted(
            (
                set(get_prg_langs_from_tickettask(self))
                & set(self.task.get_available_programming_languages())
            )
            - (
                DeprecatedContext.objects.get_deprecated_programming_languages()
                if not self.task.is_docker_checker
                else set()
            )
        )
        self.save(update_fields=["prg_lang_list"])

    def filter_unavailable_prg_lang_variants(self):
        """Same as above but prepared for variants available. WIP."""
        from codility.languages.models import DeprecatedContext

        ########################################################################
        #
        #  Change WIP to False only when variants are migrated and safe to use
        #  and you know what you are doing.
        #
        #  Otherwise it will blow up badly.
        #
        ########################################################################
        WIP = True

        programming_language_variant_name = get_programming_language_variant_name()

        available_variants = self.get_variants_available()
        if available_variants:
            available_programming_languages = set(
                available_variants.pop(programming_language_variant_name, [])
            )
        else:
            logger.warning("There are no variants for tickettask %s", self.id)
            if not WIP:
                raise Ticket.CouldNotOpen()
            return

        prg_lang_list = sorted(
            (
                available_programming_languages
                & set(self.task.get_available_programming_languages())
            )
            - (
                DeprecatedContext.objects.get_deprecated_programming_languages()
                if not self.task.is_docker_checker
                else set()
            )
        )

        prg_lang_variant = (
            {programming_language_variant_name: prg_lang_list} if prg_lang_list else {}
        )
        # TODO: The _variants... flow is currently experimental.
        #  Remove try-except clause once variants are fully integrated to the system.
        try:
            TicketTaskVariantHandler.set_variants_available_at_open(
                tickettask=self, variants={**available_variants, **prg_lang_variant}
            )
        except Exception:
            logger.exception(
                "Error in experimental variants flow while filtering unavailable programming languages"
            )

        if prg_lang_list != self.prg_lang_list:
            logger.warning(
                "Wrong list of programming languages from variants. See addtional data.",
                extra={
                    "ticket_task_id": self.id,
                    "list_from_variants": prg_lang_list,
                    "expected_prg_lang_list": self.prg_lang_list,
                },
            )

        if not prg_lang_list and WIP is False:
            raise Ticket.CouldNotOpen()

    @property
    def result_percent(self):
        """Returns the task result if there's one."""
        if self.result is None:
            return None

        # is it a valid state? self.result is not None and self.max_result is None
        if self.max_result is None:
            logger.warning("inconsistent state, there's a result but no max result")
            return None

        # Text task has max_result of 0
        if self.max_result == 0:
            return None

        return float(100 * self.result / self.max_result)

    @property
    def final_result_percent(self) -> Optional[Decimal]:
        if self.final_result is None:
            return None

        if self.final_max_result is None:
            logger.warning("inconsistent state, there's a result but no max result")
            return None

        # Text task has max_result of 0
        if self.final_max_result == 0:
            return None

        return Decimal(100 * self.final_result / self.final_max_result)

    @property
    def performance_warning(self):
        """Should we warn that performance might not be accurate?"""
        return (
            self.correctness is not None
            and self.performance is not None
            and self.correctness < 50
        )

    @property
    def performance_not_assessed(self):
        return self.correctness is not None and self.performance is None

    @memo
    def submits(self):
        return [s for s in self.ticket.submits if s.task == self.task.name]

    @memo
    def solutions(self):
        return [sol for sol in self.ticket.solutions if sol.task == self.task]

    @property
    def final_submits(self):
        return [s for s in self.submits if s.mode == "final"]

    @property
    def final_submit(self):
        fs = self.final_submits
        if fs:
            return fs[-1]
        return None

    @property
    def final_solution(self):
        for sol in self.solutions[::-1]:
            if sol.submit and sol.submit.mode == "final":
                return sol
        return None

    @memo
    def start_date(self):
        return self.ticket.get_task_start(self.task.name)

    @memo
    def close_date(self):
        return self.ticket.get_task_end(self.task.name)

    @property
    def status(self):
        started, ended = bool(self.start_date), bool(self.close_date)
        if (started, ended) == (False, False):
            return "new"
        elif (started, ended) == (True, False):
            return "open"
        elif (started, ended) == (True, True):
            return "closed"
        else:
            logger.warning(
                "weird task '%s' state, it has been closed but there is no info about opening, ticket=%s",
                self.task.name,
                (self.ticket.id),
            )
            return "closed"

    @property
    def from_previous_attempt(self):
        """Checks whether current solution is from previous attempt of the task."""
        if not self.start_date:
            # task not open yet.
            return False
        if not self.ticket.start_date and self.start_date:
            # ticket reopened but not started yet
            return True
        if self.ticket.start_date:
            return self.close_date and self.close_date < self.ticket.start_date
        return False

    @memo
    def sol_desc(self):
        return self.ticket.solution_desc(self.task.name)

    @memo
    def used_languages(self):
        return self.ticket._get_used_languages_names(self.submits, self.final_submits)

    @memo
    def final_used_language(self):
        if self.final_submit:
            return self.final_submit.prg_lang_name
        else:
            return None

    @memo
    def default_prg_lang(self):
        languages = self.ticket._get_used_languages(self.submits, self.final_submits)
        if languages:
            return languages[0]
        else:
            return None

    def save_effective_time_used(self):
        self._effective_time_used = self.effective_time_used
        self._effective_time_used_is_reliable = self.effective_time_used_is_reliable
        self.save(
            update_fields=["_effective_time_used", "_effective_time_used_is_reliable"]
        )

    @memo
    def time_used(self):
        """Total time used for task in the last reopen-attempt."""
        if not (self.start_date and self.close_date):
            return None
        if self.from_previous_attempt:
            return None

        start_date = self.start_date
        if self.ticket.start_date:
            start_date = max(start_date, self.ticket.start_date)
        return int((self.close_date - start_date).total_seconds())

    @property
    def time_used_min(self):
        return to_minutes(self.time_used)

    @memo
    def effective_time_used(self):
        """Effective time used for task in the last reopen-attempt."""
        if not (self.start_date and self.close_date):
            return None
        if self.from_previous_attempt:
            return None

        dt = timedelta(0)
        for ev1, ev2 in pairwise(self.ticket.events_list):
            # ignore events from before reopening
            if (
                self.ticket.start_date
                and ev2.timestamp <= self.ticket.start_date.replace(microsecond=0)
            ):
                continue
            if ev1.timestamp < self.start_date or ev2.timestamp > self.close_date:
                continue

            if ev1.kind in ("final", "end_task"):
                continue
            if ev1.task_name != self.task.name:
                continue
            dt += ev2.timestamp - ev1.timestamp
        return int(dt.total_seconds())

    @property
    def effective_time_used_min(self):
        return to_minutes(self._effective_time_used or self.effective_time_used)

    @memo
    def effective_time_used_is_reliable(self):
        if self.effective_time_used is None:
            return False
        if self.ticket.extended:
            return False

        # Check whether at least 90% of "effective time used"
        # has been spent in a single interval.

        intervals = [timedelta(0)]
        for ev1, ev2 in pairwise(self.ticket.events_list):
            if ev1.kind in ("final", "end_task"):
                continue
            if ev1.task_name != self.task.name:
                intervals.append(timedelta(0))
                continue
            intervals[-1] += ev2.timestamp - ev1.timestamp

        longest_interval = max(intervals).total_seconds()
        return longest_interval >= self.effective_time_used * 0.9

    @property
    def result_int(self):
        return get_result_int(self.result)

    @property
    def max_result_int(self):
        return get_result_int(self.max_result)

    @property
    def num_from1(self):
        return self.num + 1

    @property
    def task_info(self) -> TicketTaskInfo:
        return dict_remove_nones(
            TicketTaskInfo(
                name=self.task.name,
                prg_lang_list=self.prg_lang_list,
                task_weight_id=self.task_weight_id,
            )
        )

    @property
    def latest_code_snapshot(self) -> Optional[CodeSnapshot]:
        return (
            CodeSnapshot.objects.filter(ticket=self.ticket, task=self.task)
            .order_by("-timestamp")
            .first()
        )

    @property
    def task_weight(self) -> Optional[TaskWeightDataclass]:
        if not self.task_weight_id:
            return None

        from codility.campaigns.models import TaskWeight

        task_weight: Optional[TaskWeight] = TaskWeight.objects.filter(
            id=self.task_weight_id
        ).first()

        if task_weight:
            return TaskWeightDataclass(**task_weight.as_dict())

        logger.error(
            "Task weight id '%s' in ticket %s does not exist",
            self.task_weight_id,
            self.ticket.id,
        )

        return None

    def get_task_description_html(self, task_context):
        # note that custom_task_description can be {}
        # (not validated due to how JSONField works)
        if self.task.is_mcq and self.final_submit:
            assert task_context.get("human_lang", "en") == "en"
            try:
                final_submit = json.loads(self.final_submit.solution)["answers"]
            except (ValueError, KeyError):
                final_submit = None

            return render_mcq_questions(
                self.task.mcq_questions, self.task.mcq_copyright_notice, final_submit
            )
        if not self.custom_task_description:
            return self.task.get_description(task_context).content_html
        else:
            type = self.custom_task_description["type"]
            content = self.custom_task_description["content"]
            if type == "text" or type == "txt":
                template = Template(
                    """<div class="rendered-task-description">{{ content }}</div>"""
                )
                return template.render(Context({"content": content}))
            elif type == "html":
                return nh3.clean(content)
            else:
                assert False, "unknown custom_task_description type: %s" % type

    def get_representative_task_description(self):
        task_context = self.get_current_task_context() or self.clean_task_context({})
        return self.get_task_description_html(task_context)

    def get_representative_task_description_as_dict(self):
        return {"html": self.get_representative_task_description()}

    def get_task_solution_template(self, task_context: Dict[str, str]) -> Optional[str]:
        return self.task.get_solution_template(
            task_context=task_context, additional_data=self.additional_data
        )

    def get_task_example_input(self):
        return self.task.get_example_input()

    def __str__(self):
        return "%s: (%s, %s)" % (self.ticket, self.task, self.num)

    def get_current_solution(
        self, task_context: Optional[Dict[str, str]] = None
    ) -> Optional[CodeSnapshot]:
        """Get the latest solution for a given task, as a CodeSnapshot.
        If prg_lang is specified, filter by it.
        """
        assert isinstance(task_context, (type(None), dict))

        solutions = self.solutions
        if task_context:
            self.task.validate_cleaned_task_context(task_context)
            solutions = [
                sol for sol in self.solutions if sol.task_context == task_context
            ]

        if solutions:
            return solutions[-1]

        return None

    @property
    def last_result_modification(self):
        modified_results = list(self.modifiedresult_set.all())
        if modified_results:
            return max(modified_results, key=lambda x: x.create_date)
        return None

    @property
    def modified_result(self):
        result = None
        latest = self.last_result_modification
        if latest:
            if latest.result_type == ModifiedResult.RESULT_TYPES.custom:
                result = latest.result
        return result

    @property
    def modified_result_set(self):
        return self.modifiedresult_set.all()

    @property
    def final_result(self) -> Optional[Decimal]:
        final_result: Decimal = self.result

        if self.modified_result is not None:
            if self.final_max_result > 0:
                modified_result_percent: Decimal = Decimal(self.modified_result / 100)
                final_result = modified_result_percent * self.final_max_result
            else:
                final_result = self.modified_result

        return final_result

    @property
    def final_max_result(self):
        if self.max_result is not None and self.max_result != 0:
            return self.max_result
        if self.result is not None or self.modified_result is not None:
            task_weight: Optional[TaskWeightDataclass] = self.task_weight
            weight_value: int = 1

            if task_weight:
                weight_value = task_weight.value

            return 100 * weight_value
        return 0

    @property
    def task_name(self):
        return self.task.name

    @property
    def latest_uploaded_file_key(self) -> Optional[str]:
        if not self.task.allow_upload or self.final_submit is None:
            return None

        submits = self.submits
        submits.sort(key=lambda submit: submit.submit_date, reverse=True)

        for submit in submits:
            uploaded_file_key = submit.uploaded_file_key()
            if uploaded_file_key is not None:
                return uploaded_file_key

        return None

    class Meta:
        unique_together = [("ticket", "task"), ("ticket", "num")]
        ordering = ["num"]
        db_table = "tickets_task"

    def get_current_task_context(self) -> Optional[Dict[str, str]]:
        """Return the most recent task context. If there is not code snapshot, returns None."""
        current_solution = self.get_current_solution()

        if current_solution is not None:
            return current_solution.task_context

        return None

    def clean_task_context(self, task_context: Dict[str, str]) -> Dict[str, str]:
        """Return cleaned task context.

        During cleaning:
        * Any extra variants are dropped.
        * Missing variants and variants with incorrect value are replaced with default variant values.
        """
        available_variants = self.get_available_variants_with_complex_types()
        return clean_task_context(available_variants, task_context)

    def get_newest_candidate_interactions(self):
        return self.candidate_interactions.all().order_by("-create_date").first()

    def get_variants_available(self) -> Optional[Dict[str, Dict]]:
        """
        Returns a deepcopy of variants available to the tickettask at given moment.

        If ticket was created but not opened yet, copy of _variants_available_at_create will be returned.
        If ticket was opened, copy _variants_available_at_open will be returned.
        """
        available_variants = TicketTaskVariantHandler.get_variants_available_at_open(
            self
        ) or TicketTaskVariantHandler.get_variants_available_at_create(self)

        return available_variants


def get_prg_langs_from_tickettask(ticket_task: TicketTask) -> List[str]:
    prg_langs_id = Id(get_programming_language_variant_name())
    prg_langs_variant = ticket_task.get_available_variants_with_complex_types().get(
        prg_langs_id
    )
    tt_prg_langs = prg_langs_variant.values.keys() if prg_langs_variant else []
    return [str(key) for key in tt_prg_langs]


def get_lang_choices_for_docker_checker_tasks_from_ticket(
    ticket: Ticket,
) -> List[typing.Tuple[str, str]]:
    real_life_langs = []
    for tt in ticket.tickettasks:
        if tt.task.is_docker_checker:
            prg_langs_id = Id(get_programming_language_variant_name())
            prg_langs_variant = tt.get_available_variants_with_complex_types().get(
                prg_langs_id
            )
            if prg_langs_variant:
                real_life_langs += [
                    (str(value), value.fields.get("title", str(value.id)))
                    for value in prg_langs_variant.values.values()
                ]

    return real_life_langs


def get_fundamental_prg_langs_from_ticket(
    ticket: Ticket,
) -> List[str]:
    fundamental_langs = []
    for tt in ticket.tickettasks:
        if not tt.task.is_docker_checker:
            prg_langs_id = Id(get_programming_language_variant_name())
            prg_langs_variant = tt.get_available_variants_with_complex_types().get(
                prg_langs_id
            )
            if prg_langs_variant:
                fundamental_langs += (
                    prg_langs_variant.values.keys() if prg_langs_variant else []
                )

    return fundamental_langs


class StyleAssessmentRequest(models.Model):
    ticket = models.OneToOneField(Ticket, on_delete=models.CASCADE)
    request_date = models.DateTimeField(auto_now_add=True)
    requester = models.ForeignKey(User, on_delete=models.CASCADE)
    is_done = models.BooleanField(default=False)

    def __str__(self):
        return "style assessment for %s" % self.ticket_id


class TicketMailStatus(models.Model):
    ticket = models.OneToOneField(
        Ticket, on_delete=models.CASCADE, related_name="mail_status"
    )
    status = models.CharField(max_length=50, null=False)

    def __str__(self):
        return "%s: %s" % (self.ticket, self.status)


def save_ticket_mail_status(sender, **kwargs):
    event_ct = ContentType.objects.get_for_model(Ticket)

    if sender.related_to_ct_id == event_ct.id:
        ticket = sender.related_to
        (tms, created) = TicketMailStatus.objects.get_or_create(ticket=ticket)
        tms.status = sender.status_text
        tms.save()


event_signal.connect(save_ticket_mail_status)


def get_result_int(result):
    return int(round(result)) if result is not None else None


class TicketDetailShareToken(models.Model):
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)  # None = no expiry
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    content = models.CharField(max_length=64, db_index=True)

    def save(self, *args, **kwargs):
        if not self.content:
            h = hashlib.sha256()
            h.update(
                ("%s-%s-%s" % (self.ticket_id, time.time(), uuid.uuid4())).encode(
                    "ascii"
                )
            )
            self.content = h.hexdigest()
        super(TicketDetailShareToken, self).save(*args, **kwargs)

    def is_valid(self, today=None):
        if today is None:
            today = date.today()
        if self.end_date is not None:
            return self.start_date <= today <= self.end_date
        else:
            return self.start_date <= today

    def get_share_url(self):
        return reverse("ticket_detail_share", args=[self.ticket_id, self.content])

    def get_share_abs_url(self):
        return get_site_url() + self.get_share_url()


def round_result(result):
    return convert_float_to_int_rounding_up(result)


class FacadeRepository(models.Model):
    bucket_id = models.TextField()
    ticket_task_solution_id = models.CharField(max_length=255)
    ticket_task = models.ForeignKey(
        TicketTask, null=True, blank=True, on_delete=models.CASCADE
    )

    class Meta:
        indexes = [models.Index(fields=["ticket_task_solution_id"])]


class CandidateInteraction(models.Model):
    ticket_task = models.ForeignKey(
        TicketTask,
        null=True,
        blank=True,
        related_name="candidate_interactions",
        on_delete=models.CASCADE,
    )

    task_context = jsonfield.JSONField(
        null=True,
        blank=False,
        default=dict,
        help_text="Dictionary of task variants (e.g. prg_lang, human_lang)",
    )

    create_date: datetime = models.DateTimeField(
        null=False,
        blank=False,
        default=datetime.utcnow,
        help_text="Date when the entity is created",
    )

    class Meta:
        indexes = [models.Index(fields=["create_date"])]


class IntroRegistrationHeader(enum.Enum):
    """Used to figure out which registration section header to show."""

    DEMO = "demo"
    LEGACY = "legacy"
    LESSON = "lesson"
    TRAINING = "training"
    CHALLENGE = "challenge"


class AbstractIntro(abc.ABC):
    # Intro context helpers
    @abc.abstractmethod
    def get_registration_header(self) -> IntroRegistrationHeader:
        ...

    def is_demo(self) -> bool:
        return False

    def is_training(self) -> bool:
        return False

    def is_current_challenge(self) -> bool:
        return False

    def allow_registration_skip(self) -> bool:
        return False

    def show_before_you_begin(self) -> bool:
        return False

    def show_demo_button(self) -> bool:
        return False

    def show_help(self) -> bool:
        return False

    # END intro context helpers.

    @abc.abstractmethod
    def get_timelimit_min(self) -> int:
        ...

    @abc.abstractmethod
    def get_programming_languages(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_task_name(self) -> str:
        ...

    @abc.abstractmethod
    def get_ticket_origin(self) -> str:
        ...

    def create_ticket(
        self,
        creator: Optional[User] = None,
        candidate: Optional[User] = None,
        limit_variants: Optional[Dict[str, List[str]]] = None,
    ) -> Ticket:
        """Creates a ticket for candidate with a creator.

        If `creator` is not a programmer, the resulting Ticket won't have any creator
        assigned.
        If `limit_variants` is not None, it will be used to limit the variants available
        for the given ticket. The variants available will be a common subset of both the
        argument value and variant values available to the task.
        Currently, only programming languages are supported.
        """
        from codility.tickets.utils.ticket_builder import TicketBuilder

        ticket_creator = creator if creator and creator.is_programmer else None
        prg_lang_list = set(self.get_programming_languages())
        if limit_variants:
            limit_programming_languages = limit_variants.get(
                get_programming_language_variant_name(), []
            )
            if limit_programming_languages:
                prg_lang_list &= set(limit_programming_languages)

        # If there are no valid programming languages, default to the intro's.
        if not prg_lang_list:
            prg_lang_list = self.get_programming_languages()

        t = TicketBuilder(
            data=[
                {
                    "extra_prefix": self.get_ticket_origin(),
                    "origin": self.get_ticket_origin(),
                    "task_infos": [
                        {
                            "name": self.get_task_name(),
                            "prg_lang_list": list(prg_lang_list),
                        }
                    ],
                }
            ],
            creator=ticket_creator,
        ).create()[0]

        t.timelimit = self.get_timelimit_min() * 60
        return t


RECOMMENDATION_CHOICES = [
    ("thumbs-up", "thumbs-up"),
    ("thumbs-down", "thumbs-down"),
    ("star", "star"),
]


class Review(models.Model):
    create_date = models.DateTimeField(auto_now_add=True)
    update_date = models.DateTimeField(auto_now=True)
    ticket = models.ForeignKey(Ticket, related_name="reviews", on_delete=models.CASCADE)
    reviewer = models.ForeignKey(User, related_name="reviews", on_delete=models.CASCADE)
    recommendation = models.CharField(max_length=16, choices=RECOMMENDATION_CHOICES)
    feedback = models.TextField(blank=True, null=True)

    def get_text_representation(self) -> str:
        recommendation_emojis = {
            "thumbs-up": "\U0001F44D",
            "thumbs-down": "\U0001F44E",
            "star": "\U00002B50",
        }
        recommendation_phrases = {
            "thumbs-up": "Recommended",
            "thumbs-down": "Not recommended",
            "star": "Definitely recommended",
        }
        optional_feedback = f": {self.feedback}" if self.feedback else ""
        return (
            f"{recommendation_emojis.get(self.recommendation)} {recommendation_phrases.get(self.recommendation)} "
            f"by {self.reviewer.get_full_name()}{optional_feedback}"
        )


class ReviewAssignment(models.Model):
    ticket_id = models.CharField(max_length=16, primary_key=True)
    user_id = models.IntegerField(db_index=True)
    create_date = models.DateTimeField(auto_now_add=True)


class ReviewStatus(models.Model):
    ticket_id = models.CharField(max_length=16, primary_key=True)
    ignored = models.BooleanField()


class TicketReviewStatusChecker:
    @staticmethod
    def get_review_status(ticket: Ticket) -> str:
        if Review.objects.filter(ticket=ticket).exists():
            return "reviewed"
        if ReviewStatus.objects.filter(ticket_id=ticket.id, ignored=True).exists():
            return "skip-review"
        if ReviewAssignment.objects.filter(ticket_id=ticket.id).exists():
            return "assigned"
        return "not-applicable"

    @staticmethod
    def filter_tickets_up_for_review(ticket_qs: QuerySet) -> QuerySet:
        ignored_statuses = ReviewStatus.objects.filter(
            ticket_id__in=(ticket_qs.values_list("id", flat=True)), ignored=True
        )
        return ticket_qs.exclude(
            id__in=(ignored_statuses.values_list("ticket_id", flat=True))
        )

    @staticmethod
    def filter_tickets_skip_review(ticket_qs: QuerySet) -> QuerySet:
        ignored_statuses = ReviewStatus.objects.filter(
            ticket_id__in=(ticket_qs.values_list("id", flat=True)), ignored=True
        )
        return ticket_qs.filter(
            id__in=(ignored_statuses.values_list("ticket_id", flat=True))
        ).distinct()

    @staticmethod
    def filter_by_review_status(tickets_qs: QuerySet, review_status: str) -> QuerySet:
        codelive_tickets_q: Q = Q(is_codelive=True)
        testdrives_tickets_q: Q = Q(origin="try")
        training_tickets_q: Q = Q(origin="training")
        reviewed_tickets_q: Q = Q(reviews__isnull=False)
        not_completed_tickets_q: Q = Q(close_date__isnull=True)

        tickets_review_can_be_skipped: QuerySet = tickets_qs.exclude(
            testdrives_tickets_q
            | reviewed_tickets_q
            | codelive_tickets_q
            | training_tickets_q
        )

        tickets_can_be_reviewed: QuerySet = tickets_review_can_be_skipped.exclude(
            not_completed_tickets_q
        )

        if review_status == "reviewed":
            return tickets_qs.filter(reviewed_tickets_q).distinct()
        if review_status == "assigned":
            return TicketReviewStatusChecker.filter_tickets_up_for_review(
                tickets_can_be_reviewed
            )
        if review_status == "skip-review":
            return TicketReviewStatusChecker.filter_tickets_skip_review(
                tickets_review_can_be_skipped
            )
        else:
            return tickets_qs
