const { body, validationResult } = require('express-validator');

/**
 * Middleware to handle validation errors
 */
function handleValidationErrors(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      message: 'Validation failed',
      errors: errors.array().map(err => ({
        field: err.path,
        message: err.msg
      }))
    });
  }
  next();
}

/**
 * Validation rules for user registration
 */
const validateRegistration = [
  body('email')
    .trim()
    .isEmail().withMessage('Please enter a valid email')
    .normalizeEmail(),

  body('password')
    .isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),

  body('firstName')
    .trim()
    .notEmpty().withMessage('First name is required')
    .isLength({ max: 50 }).withMessage('First name must be less than 50 characters')
    .escape(),

  body('lastName')
    .trim()
    .notEmpty().withMessage('Last name is required')
    .isLength({ max: 50 }).withMessage('Last name must be less than 50 characters')
    .escape(),

  handleValidationErrors
];

/**
 * Validation rules for user login
 */
const validateLogin = [
  body('email')
    .trim()
    .isEmail().withMessage('Please enter a valid email')
    .normalizeEmail(),

  body('password')
    .notEmpty().withMessage('Password is required'),

  handleValidationErrors
];

/**
 * Validation rules for assessment creation/update
 */
const validateAssessment = [
  body('employeeName')
    .trim()
    .notEmpty().withMessage('Employee name is required')
    .isLength({ min: 1, max: 100 }).withMessage('Employee name must be between 1 and 100 characters')
    .escape(),

  body('assessmentDate')
    .optional()
    .isISO8601().withMessage('Assessment date must be a valid date'),

  // Validate all 19 metrics (0-5 range)
  body('metrics.sharedVision').isInt({ min: 0, max: 5 }).withMessage('Shared Vision must be between 0 and 5'),
  body('metrics.strategy').isInt({ min: 0, max: 5 }).withMessage('Strategy must be between 0 and 5'),
  body('metrics.businessAlignment').isInt({ min: 0, max: 5 }).withMessage('Business Alignment must be between 0 and 5'),
  body('metrics.customerFocus').isInt({ min: 0, max: 5 }).withMessage('Customer Focus must be between 0 and 5'),

  body('metrics.crossFunctionalTeams').isInt({ min: 0, max: 5 }).withMessage('Cross-Functional Teams must be between 0 and 5'),
  body('metrics.clarityInPriorities').isInt({ min: 0, max: 5 }).withMessage('Clarity in Priorities must be between 0 and 5'),
  body('metrics.acceptanceCriteria').isInt({ min: 0, max: 5 }).withMessage('Acceptance Criteria must be between 0 and 5'),
  body('metrics.enablingFocus').isInt({ min: 0, max: 5 }).withMessage('Enabling Focus must be between 0 and 5'),
  body('metrics.engagement').isInt({ min: 0, max: 5 }).withMessage('Engagement must be between 0 and 5'),

  body('metrics.feedback').isInt({ min: 0, max: 5 }).withMessage('Feedback must be between 0 and 5'),
  body('metrics.enablingAutonomy').isInt({ min: 0, max: 5 }).withMessage('Enabling Autonomy must be between 0 and 5'),
  body('metrics.changeAndAmbiguity').isInt({ min: 0, max: 5 }).withMessage('Change and Ambiguity must be between 0 and 5'),
  body('metrics.desiredCulture').isInt({ min: 0, max: 5 }).withMessage('Desired Culture must be between 0 and 5'),
  body('metrics.workAutonomously').isInt({ min: 0, max: 5 }).withMessage('Works Autonomously must be between 0 and 5'),

  body('metrics.stakeholders').isInt({ min: 0, max: 5 }).withMessage('Stakeholders must be between 0 and 5'),
  body('metrics.teamAttrition').isInt({ min: 0, max: 5 }).withMessage('Team Attrition must be between 0 and 5'),
  body('metrics.teams').isInt({ min: 0, max: 5 }).withMessage('Teams must be between 0 and 5'),
  body('metrics.developingPeople').isInt({ min: 0, max: 5 }).withMessage('Developing People must be between 0 and 5'),
  body('metrics.subordinatesForSuccess').isInt({ min: 0, max: 5 }).withMessage('Subordinates for Success must be between 0 and 5'),

  handleValidationErrors
];

/**
 * Validation rules for user update (self or admin)
 */
const validateUserUpdate = [
  body('email')
    .optional()
    .trim()
    .isEmail().withMessage('Please enter a valid email')
    .normalizeEmail(),

  body('password')
    .optional()
    .isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),

  body('firstName')
    .optional()
    .trim()
    .isLength({ max: 50 }).withMessage('First name must be less than 50 characters')
    .escape(),

  body('lastName')
    .optional()
    .trim()
    .isLength({ max: 50 }).withMessage('Last name must be less than 50 characters')
    .escape(),

  body('role')
    .optional()
    .isIn(['assessor', 'admin']).withMessage('Role must be assessor or admin'),

  body('currentPassword')
    .optional()
    .notEmpty().withMessage('Current password cannot be empty'),

  handleValidationErrors
];

/**
 * Validation rules for user creation (admin only)
 */
const validateUserCreate = [
  body('email')
    .trim()
    .isEmail().withMessage('Please enter a valid email')
    .normalizeEmail(),

  body('password')
    .isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),

  body('firstName')
    .trim()
    .notEmpty().withMessage('First name is required')
    .isLength({ max: 50 }).withMessage('First name must be less than 50 characters')
    .escape(),

  body('lastName')
    .trim()
    .notEmpty().withMessage('Last name is required')
    .isLength({ max: 50 }).withMessage('Last name must be less than 50 characters')
    .escape(),

  body('role')
    .optional()
    .isIn(['assessor', 'admin']).withMessage('Role must be assessor or admin'),

  handleValidationErrors
];

module.exports = {
  validateRegistration,
  validateLogin,
  validateAssessment,
  validateUserUpdate,
  validateUserCreate,
  handleValidationErrors
};
