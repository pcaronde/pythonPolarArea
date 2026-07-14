const express = require('express');
const router = express.Router();
const rateLimit = require('express-rate-limit');
const {
  register,
  login,
  getCurrentUser,
  getAuthConfig,
  logout,
  getAllUsers,
  createUser,
  updateUser,
  deleteUser
} = require('../controllers/authController');
const {
  validateRegistration,
  validateLogin,
  validateUserUpdate,
  validateUserCreate
} = require('../middleware/validation');
const { authenticateToken, requireRole } = require('../middleware/authMiddleware');

// Rate limiter for auth endpoints (prevent brute force)
// Skip rate limiting in test environment
const authLimiter = process.env.NODE_ENV === 'test'
  ? (req, res, next) => next()
  : rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 5, // 5 requests per window
      message: 'Too many authentication attempts, please try again later',
      standardHeaders: true,
      legacyHeaders: false
    });

/**
 * GET /api/auth/config
 * Get authentication configuration (public endpoint)
 * Returns: { allowRegistration }
 */
router.get('/config', getAuthConfig);

/**
 * POST /api/auth/register
 * Register a new user
 * Body: { email, password, firstName, lastName }
 */
router.post('/register', authLimiter, validateRegistration, register);

/**
 * POST /api/auth/login
 * Login with email and password
 * Body: { email, password }
 * Returns: { token, user }
 */
router.post('/login', authLimiter, validateLogin, login);

/**
 * GET /api/auth/me
 * Get current logged-in user info
 * Requires: Authorization header with Bearer token
 * Returns: { user }
 */
router.get('/me', authenticateToken, getCurrentUser);

/**
 * POST /api/auth/logout
 * Logout (client-side token removal)
 * Requires: Authorization header with Bearer token
 */
router.post('/logout', authenticateToken, logout);

// ==================== User Management Routes ====================

/**
 * GET /api/auth/users
 * Get all users (admin only)
 * Requires: Authorization header with Bearer token, admin role
 */
router.get('/users', authenticateToken, requireRole('admin'), getAllUsers);

/**
 * POST /api/auth/users
 * Create a new user (admin only)
 * Body: { email, password, firstName, lastName, role? }
 * Requires: Authorization header with Bearer token, admin role
 */
router.post('/users', authenticateToken, requireRole('admin'), validateUserCreate, createUser);

/**
 * PUT /api/auth/users/:id
 * Update a user (self or admin)
 * Body: { email?, password?, firstName?, lastName?, role?, currentPassword? }
 * Requires: Authorization header with Bearer token
 */
router.put('/users/:id', authenticateToken, validateUserUpdate, updateUser);

/**
 * DELETE /api/auth/users/:id
 * Delete a user (admin only)
 * Requires: Authorization header with Bearer token, admin role
 */
router.delete('/users/:id', authenticateToken, requireRole('admin'), deleteUser);

module.exports = router;
