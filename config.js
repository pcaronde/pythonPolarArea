// Configuration for Performance Assessment
export const ASSESSMENT_CONFIG = {
    ratingScale: {
        min: 0,
        max: 5,
        labels: {
            0: 'Not Applicable',
            1: 'Very Poor',
            2: 'Poor',
            3: 'Fair',
            4: 'Good',
            5: 'Excellent'
        }
    },

    themes: {
        'Strategic Vision': {
            color: 'rgba(255, 99, 132, %a)',
            metrics: [
                { id: 'sharedVision', label: 'Shared Vision' },
                { id: 'strategy', label: 'Strategy' },
                { id: 'businessAlignment', label: 'Business Alignment' },
                { id: 'customerFocus', label: 'Customer Focus' }
            ]
        },
        'Focus and Engagement': {
            color: 'rgba(54, 162, 235, %a)',
            metrics: [
                { id: 'crossFunctionalTeams', label: 'Cross-Functional Teams' },
                { id: 'clarityInPriorities', label: 'Clarity in Priorities' },
                { id: 'acceptanceCriteria', label: 'Acceptance Criteria' },
                { id: 'enablingFocus', label: 'Enabling Focus' },
                { id: 'engagement', label: 'Engagement' }
            ]
        },
        'Autonomy and Change': {
            color: 'rgba(255, 206, 86, %a)',
            metrics: [
                { id: 'feedback', label: 'Feedback' },
                { id: 'enablingAutonomy', label: 'Enabling Autonomy' },
                { id: 'changeAndAmbiguity', label: 'Change and Ambiguity' },
                { id: 'desiredCulture', label: 'Desired Culture' },
                { id: 'workAutonomously', label: 'Works Autonomously' }
            ]
        },
        'Stakeholders and Team': {
            color: 'rgba(75, 192, 192, %a)',
            metrics: [
                { id: 'stakeholders', label: 'Stakeholders' },
                { id: 'teamAttrition', label: 'Team Attrition' },
                { id: 'teams', label: 'Teams' },
                { id: 'developingPeople', label: 'Developing People' },
                { id: 'subordinatesForSuccess', label: 'Subordinates for Success' }
            ]
        }
    }
};
