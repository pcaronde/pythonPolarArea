/**
 * Authentication API Tests
 * Tests for login, register, and auth endpoints
 */

const request = require('supertest');
const express = require('express');
const User = require('../models/User');
const authRoutes = require('../routes/auth');
const { createTestUser, createAdminUser } = require('./helpers');

// Create Express app for testing (without rate limiting for tests)
const app = express();
app.use(express.json());
app.use('/api/auth', authRoutes);

describe('Authentication API', () => {
  describe('POST /api/auth/register', () => {
    beforeEach(() => {
      process.env.ALLOW_REGISTRATION = 'true';
    });

    it('should register a new user', async () => {
      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          password: 'password123',
          firstName: 'New',
          lastName: 'User'
        })
        .expect(201);

      expect(res.body.message).toBe('User registered successfully');
      expect(res.body.token).toBeDefined();
      expect(res.body.user.email).toBe('newuser@example.com');
      expect(res.body.user.role).toBe('assessor');
      expect(res.body.user).not.toHaveProperty('password');
    });

    it('should reject registration when disabled', async () => {
      process.env.ALLOW_REGISTRATION = 'false';

      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          password: 'password123',
          firstName: 'New',
          lastName: 'User'
        })
        .expect(403);

      expect(res.body.message).toContain('disabled');
    });

    it('should reject duplicate email', async () => {
      await createTestUser({ email: 'existing@example.com' });

      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'existing@example.com',
          password: 'password123',
          firstName: 'Duplicate',
          lastName: 'User'
        })
        .expect(409);

      expect(res.body.message).toBe('Email already registered');
    });

    it('should validate email format', async () => {
      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'invalid-email',
          password: 'password123',
          firstName: 'Test',
          lastName: 'User'
        })
        .expect(400);

      expect(res.body.message).toBe('Validation failed');
    });

    it('should validate password length', async () => {
      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'test@example.com',
          password: '12345',
          firstName: 'Test',
          lastName: 'User'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'password' })
      );
    });

    it('should require all fields', async () => {
      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'test@example.com'
        })
        .expect(400);

      expect(res.body.message).toBe('Validation failed');
    });
  });

  describe('POST /api/auth/login', () => {
    it('should login with valid credentials', async () => {
      await createTestUser({
        email: 'login@example.com',
        password: 'password123'
      });

      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'login@example.com',
          password: 'password123'
        })
        .expect(200);

      expect(res.body.message).toBe('Login successful');
      expect(res.body.token).toBeDefined();
      expect(res.body.user.email).toBe('login@example.com');
    });

    it('should reject invalid password', async () => {
      await createTestUser({
        email: 'login@example.com',
        password: 'password123'
      });

      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'login@example.com',
          password: 'wrongpassword'
        })
        .expect(401);

      expect(res.body.message).toBe('Invalid credentials');
    });

    it('should reject non-existent user', async () => {
      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: 'password123'
        })
        .expect(401);

      expect(res.body.message).toBe('Invalid credentials');
    });

    it('should validate email format', async () => {
      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'invalid-email',
          password: 'password123'
        })
        .expect(400);

      expect(res.body.message).toBe('Validation failed');
    });

    it('should update lastLogin on successful login', async () => {
      const { user } = await createTestUser({
        email: 'lastlogin@example.com',
        password: 'password123'
      });

      const originalLastLogin = user.lastLogin;

      await request(app)
        .post('/api/auth/login')
        .send({
          email: 'lastlogin@example.com',
          password: 'password123'
        })
        .expect(200);

      const updatedUser = await User.findById(user._id);
      expect(updatedUser.lastLogin).not.toEqual(originalLastLogin);
    });
  });

  describe('GET /api/auth/me', () => {
    it('should return current user info', async () => {
      const { user, token } = await createTestUser();

      const res = await request(app)
        .get('/api/auth/me')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(res.body.user.email).toBe(user.email);
      expect(res.body.user.firstName).toBe(user.firstName);
      expect(res.body.user.role).toBe(user.role);
      expect(res.body.user).not.toHaveProperty('password');
    });

    it('should return 401 without token', async () => {
      const res = await request(app)
        .get('/api/auth/me')
        .expect(401);

      expect(res.body.message).toBe('Access token required');
    });

    it('should return 403 with invalid token', async () => {
      const res = await request(app)
        .get('/api/auth/me')
        .set('Authorization', 'Bearer invalid-token')
        .expect(403);

      expect(res.body.message).toBe('Invalid token');
    });
  });

  describe('POST /api/auth/logout', () => {
    it('should logout successfully', async () => {
      const { token } = await createTestUser();

      const res = await request(app)
        .post('/api/auth/logout')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(res.body.message).toBe('Logged out successfully');
    });

    it('should return 401 without token', async () => {
      await request(app)
        .post('/api/auth/logout')
        .expect(401);
    });
  });

  describe('GET /api/auth/config', () => {
    it('should return registration config when enabled', async () => {
      process.env.ALLOW_REGISTRATION = 'true';

      const res = await request(app)
        .get('/api/auth/config')
        .expect(200);

      expect(res.body.allowRegistration).toBe(true);
    });

    it('should return registration config when disabled', async () => {
      process.env.ALLOW_REGISTRATION = 'false';

      const res = await request(app)
        .get('/api/auth/config')
        .expect(200);

      expect(res.body.allowRegistration).toBe(false);
    });
  });
});
