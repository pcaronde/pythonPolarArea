/**
 * JWT Config Tests
 * Tests for token generation and verification
 */

const jwt = require('jsonwebtoken');
const { generateToken, verifyToken, JWT_SECRET } = require('../config/jwt');

describe('JWT Config', () => {
  describe('generateToken', () => {
    it('should produce a token containing the given payload', () => {
      const token = generateToken({ userId: 'abc123', email: 'a@example.com', role: 'assessor' });
      const decoded = jwt.decode(token);

      expect(decoded.userId).toBe('abc123');
      expect(decoded.email).toBe('a@example.com');
      expect(decoded.role).toBe('assessor');
    });
  });

  describe('verifyToken', () => {
    it('should return the decoded payload for a valid token', () => {
      const token = generateToken({ userId: 'abc123', email: 'a@example.com', role: 'assessor' });
      const payload = verifyToken(token);

      expect(payload.userId).toBe('abc123');
      expect(payload.email).toBe('a@example.com');
    });

    it('should throw JsonWebTokenError for a tampered token', () => {
      const token = generateToken({ userId: 'abc123' });
      const tampered = token.slice(0, -1) + (token.slice(-1) === 'a' ? 'b' : 'a');

      expect(() => verifyToken(tampered)).toThrow(jwt.JsonWebTokenError);
    });

    it('should throw JsonWebTokenError for a token signed with a different secret', () => {
      const token = jwt.sign({ userId: 'abc123' }, 'wrong-secret');

      expect(() => verifyToken(token)).toThrow(jwt.JsonWebTokenError);
    });

    it('should throw TokenExpiredError for an expired token', () => {
      const token = jwt.sign({ userId: 'abc123' }, JWT_SECRET, { expiresIn: -1 });

      expect(() => verifyToken(token)).toThrow(jwt.TokenExpiredError);
    });
  });
});
