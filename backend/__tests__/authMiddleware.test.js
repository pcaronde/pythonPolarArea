/**
 * Auth Middleware Tests
 * Tests for authenticateToken and requireRole
 */

const jwt = require('jsonwebtoken');
const { generateToken, JWT_SECRET } = require('../config/jwt');
const { authenticateToken, requireRole } = require('../middleware/authMiddleware');
const { createTestUser } = require('./helpers');

function mockRes() {
  const res = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res;
}

describe('authMiddleware', () => {
  describe('authenticateToken', () => {
    it('should return 401 when no Authorization header is present', async () => {
      const req = { headers: {} };
      const res = mockRes();
      const next = jest.fn();

      await authenticateToken(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({ message: 'Access token required' });
      expect(next).not.toHaveBeenCalled();
    });

    it('should return 401 when header has no token after Bearer', async () => {
      const req = { headers: { authorization: 'Bearer' } };
      const res = mockRes();
      const next = jest.fn();

      await authenticateToken(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(next).not.toHaveBeenCalled();
    });

    it('should return 403 Invalid token for a malformed JWT', async () => {
      const req = { headers: { authorization: 'Bearer not-a-real-token' } };
      const res = mockRes();
      const next = jest.fn();

      await authenticateToken(req, res, next);

      expect(res.status).toHaveBeenCalledWith(403);
      expect(res.json).toHaveBeenCalledWith({ message: 'Invalid token' });
      expect(next).not.toHaveBeenCalled();
    });

    it('should return 403 Token expired for an expired JWT', async () => {
      const expired = jwt.sign({ userId: 'abc123' }, JWT_SECRET, { expiresIn: -1 });
      const req = { headers: { authorization: `Bearer ${expired}` } };
      const res = mockRes();
      const next = jest.fn();

      await authenticateToken(req, res, next);

      expect(res.status).toHaveBeenCalledWith(403);
      expect(res.json).toHaveBeenCalledWith({ message: 'Token expired' });
      expect(next).not.toHaveBeenCalled();
    });

    it('should return 401 User not found when the token user no longer exists', async () => {
      const token = generateToken({ userId: '507f1f77bcf86cd799439011', email: 'ghost@example.com', role: 'assessor' });
      const req = { headers: { authorization: `Bearer ${token}` } };
      const res = mockRes();
      const next = jest.fn();

      await authenticateToken(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({ message: 'User not found' });
      expect(next).not.toHaveBeenCalled();
    });

    it('should attach userId, userEmail, userRole, and user, then call next() for a valid token', async () => {
      const { user, token } = await createTestUser();
      const req = { headers: { authorization: `Bearer ${token}` } };
      const res = mockRes();
      const next = jest.fn();

      await authenticateToken(req, res, next);

      expect(req.userId).toBe(String(user._id));
      expect(req.userEmail).toBe(user.email);
      expect(req.userRole).toBe(user.role);
      expect(req.user._id.toString()).toBe(user._id.toString());
      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });
  });

  describe('requireRole', () => {
    it('should return 403 when userRole is missing', () => {
      const req = {};
      const res = mockRes();
      const next = jest.fn();

      requireRole('admin')(req, res, next);

      expect(res.status).toHaveBeenCalledWith(403);
      expect(res.json).toHaveBeenCalledWith({ message: 'Insufficient permissions' });
      expect(next).not.toHaveBeenCalled();
    });

    it('should return 403 when userRole is not in the allowed list', () => {
      const req = { userRole: 'assessor' };
      const res = mockRes();
      const next = jest.fn();

      requireRole('admin')(req, res, next);

      expect(res.status).toHaveBeenCalledWith(403);
      expect(next).not.toHaveBeenCalled();
    });

    it('should call next() when userRole matches an allowed role', () => {
      const req = { userRole: 'admin' };
      const res = mockRes();
      const next = jest.fn();

      requireRole('admin', 'assessor')(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });
  });
});
