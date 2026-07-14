/**
 * Validation Middleware Tests
 * Tests for input validation
 */

const request = require('supertest');
const express = require('express');
const {
  validateRegistration,
  validateLogin,
  validateUserUpdate,
  validateUserCreate
} = require('../middleware/validation');

// Create test app
function createTestApp(validator, handler) {
  const app = express();
  app.use(express.json());
  app.post('/test', validator, handler || ((req, res) => res.json({ success: true })));
  return app;
}

describe('Validation Middleware', () => {
  describe('validateRegistration', () => {
    const app = createTestApp(validateRegistration);

    it('should pass with valid data', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123',
          firstName: 'Test',
          lastName: 'User'
        })
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should fail with invalid email', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'not-an-email',
          password: 'password123',
          firstName: 'Test',
          lastName: 'User'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'email' })
      );
    });

    it('should fail with short password', async () => {
      const res = await request(app)
        .post('/test')
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

    it('should fail with missing firstName', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123',
          lastName: 'User'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'firstName' })
      );
    });

    it('should fail with missing lastName', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123',
          firstName: 'Test'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'lastName' })
      );
    });

    it('should fail with firstName too long', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123',
          firstName: 'A'.repeat(51),
          lastName: 'User'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'firstName' })
      );
    });

    it('should normalize email to lowercase', async () => {
      const testApp = express();
      testApp.use(express.json());
      testApp.post('/test', validateRegistration, (req, res) => {
        res.json({ email: req.body.email });
      });

      const res = await request(testApp)
        .post('/test')
        .send({
          email: 'TEST@EXAMPLE.COM',
          password: 'password123',
          firstName: 'Test',
          lastName: 'User'
        })
        .expect(200);

      expect(res.body.email).toBe('test@example.com');
    });
  });

  describe('validateLogin', () => {
    const app = createTestApp(validateLogin);

    it('should pass with valid data', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123'
        })
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should fail with invalid email', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'not-an-email',
          password: 'password123'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'email' })
      );
    });

    it('should fail with missing password', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'password' })
      );
    });
  });

  describe('validateUserUpdate', () => {
    const app = createTestApp(validateUserUpdate);

    it('should pass with no data (all optional)', async () => {
      const res = await request(app)
        .post('/test')
        .send({})
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should pass with valid email', async () => {
      const res = await request(app)
        .post('/test')
        .send({ email: 'new@example.com' })
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should fail with invalid email', async () => {
      const res = await request(app)
        .post('/test')
        .send({ email: 'not-an-email' })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'email' })
      );
    });

    it('should pass with valid password', async () => {
      const res = await request(app)
        .post('/test')
        .send({ password: 'newpassword123' })
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should fail with short password', async () => {
      const res = await request(app)
        .post('/test')
        .send({ password: '12345' })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'password' })
      );
    });

    it('should pass with valid role', async () => {
      const res = await request(app)
        .post('/test')
        .send({ role: 'admin' })
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should fail with invalid role', async () => {
      const res = await request(app)
        .post('/test')
        .send({ role: 'superuser' })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'role' })
      );
    });

    it('should fail with firstName too long', async () => {
      const res = await request(app)
        .post('/test')
        .send({ firstName: 'A'.repeat(51) })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'firstName' })
      );
    });
  });

  describe('validateUserCreate', () => {
    const app = createTestApp(validateUserCreate);

    it('should pass with valid data', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123',
          firstName: 'Test',
          lastName: 'User'
        })
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should pass with valid role', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123',
          firstName: 'Test',
          lastName: 'User',
          role: 'admin'
        })
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should fail with invalid role', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com',
          password: 'password123',
          firstName: 'Test',
          lastName: 'User',
          role: 'invalid'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'role' })
      );
    });

    it('should fail with missing required fields', async () => {
      const res = await request(app)
        .post('/test')
        .send({
          email: 'test@example.com'
        })
        .expect(400);

      expect(res.body.errors.length).toBeGreaterThan(0);
    });
  });
});
