/**
 * User Management API Tests
 * Tests for user CRUD operations
 */

const request = require('supertest');
const express = require('express');
const mongoose = require('mongoose');
const User = require('../models/User');
const Assessment = require('../models/Assessment');
const authRoutes = require('../routes/auth');
const { createTestUser, createAdminUser, createTestUsers } = require('./helpers');

// Create Express app for testing
const app = express();
app.use(express.json());
app.use('/api/auth', authRoutes);

describe('User Management API', () => {
  describe('GET /api/auth/users', () => {
    it('should return 401 without authentication', async () => {
      const res = await request(app)
        .get('/api/auth/users')
        .expect(401);

      expect(res.body.message).toBe('Access token required');
    });

    it('should return 403 for non-admin users', async () => {
      const { token } = await createTestUser();

      const res = await request(app)
        .get('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .expect(403);

      expect(res.body.message).toBe('Insufficient permissions');
    });

    it('should return all users for admin', async () => {
      const { token } = await createAdminUser();
      await createTestUsers(3);

      const res = await request(app)
        .get('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(res.body.users).toHaveLength(4); // admin + 3 users
      expect(res.body.users[0]).not.toHaveProperty('password');
    });
  });

  describe('POST /api/auth/users', () => {
    it('should return 401 without authentication', async () => {
      const res = await request(app)
        .post('/api/auth/users')
        .send({
          email: 'new@example.com',
          password: 'password123',
          firstName: 'New',
          lastName: 'User'
        })
        .expect(401);
    });

    it('should return 403 for non-admin users', async () => {
      const { token } = await createTestUser();

      const res = await request(app)
        .post('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'new@example.com',
          password: 'password123',
          firstName: 'New',
          lastName: 'User'
        })
        .expect(403);
    });

    it('should create user for admin', async () => {
      const { token } = await createAdminUser();

      const res = await request(app)
        .post('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'newuser@example.com',
          password: 'password123',
          firstName: 'New',
          lastName: 'User',
          role: 'assessor'
        })
        .expect(201);

      expect(res.body.message).toBe('User created successfully');
      expect(res.body.user.email).toBe('newuser@example.com');
      expect(res.body.user.role).toBe('assessor');
      expect(res.body.user).not.toHaveProperty('password');
    });

    it('should create admin user when specified', async () => {
      const { token } = await createAdminUser();

      const res = await request(app)
        .post('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'newadmin@example.com',
          password: 'password123',
          firstName: 'New',
          lastName: 'Admin',
          role: 'admin'
        })
        .expect(201);

      expect(res.body.user.role).toBe('admin');
    });

    it('should reject duplicate email', async () => {
      const { token } = await createAdminUser({ email: 'existing@example.com' });

      const res = await request(app)
        .post('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'existing@example.com',
          password: 'password123',
          firstName: 'Duplicate',
          lastName: 'User'
        })
        .expect(409);

      expect(res.body.message).toBe('Email already registered');
    });

    it('should validate required fields', async () => {
      const { token } = await createAdminUser();

      const res = await request(app)
        .post('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'invalid-email'
        })
        .expect(400);

      expect(res.body.message).toBe('Validation failed');
    });

    it('should validate password length', async () => {
      const { token } = await createAdminUser();

      const res = await request(app)
        .post('/api/auth/users')
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'test@example.com',
          password: '123',
          firstName: 'Test',
          lastName: 'User'
        })
        .expect(400);

      expect(res.body.errors).toContainEqual(
        expect.objectContaining({ field: 'password' })
      );
    });
  });

  describe('PUT /api/auth/users/:id', () => {
    it('should allow user to update own profile', async () => {
      const { user, token } = await createTestUser();

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          firstName: 'Updated',
          lastName: 'Name'
        })
        .expect(200);

      expect(res.body.user.firstName).toBe('Updated');
      expect(res.body.user.lastName).toBe('Name');
    });

    it('should require current password for email change', async () => {
      const { user, token } = await createTestUser();

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'newemail@example.com'
        })
        .expect(400);

      expect(res.body.message).toBe('Current password required for this change');
    });

    it('should require current password for password change', async () => {
      const { user, token } = await createTestUser();

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          password: 'newpassword123'
        })
        .expect(400);

      expect(res.body.message).toBe('Current password required for this change');
    });

    it('should update email with correct current password', async () => {
      const { user, token } = await createTestUser({ password: 'password123' });

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'newemail@example.com',
          currentPassword: 'password123'
        })
        .expect(200);

      expect(res.body.user.email).toBe('newemail@example.com');
    });

    it('should reject wrong current password', async () => {
      const { user, token } = await createTestUser({ password: 'password123' });

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          email: 'newemail@example.com',
          currentPassword: 'wrongpassword'
        })
        .expect(401);

      expect(res.body.message).toBe('Current password is incorrect');
    });

    it('should prevent non-admin from changing role', async () => {
      const { user, token } = await createTestUser();

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          role: 'admin'
        })
        .expect(403);

      expect(res.body.message).toBe('Only admins can change user roles');
    });

    it('should allow admin to change another user role', async () => {
      const { token: adminToken } = await createAdminUser();
      const { user } = await createTestUser();

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          role: 'admin'
        })
        .expect(200);

      expect(res.body.user.role).toBe('admin');
    });

    it('should allow admin to update any user without current password', async () => {
      const { token: adminToken } = await createAdminUser();
      const { user } = await createTestUser();

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          firstName: 'AdminUpdated',
          email: 'adminupdated@example.com'
        })
        .expect(200);

      expect(res.body.user.firstName).toBe('AdminUpdated');
      expect(res.body.user.email).toBe('adminupdated@example.com');
    });

    it('should prevent user from editing another user', async () => {
      const { token } = await createTestUser();
      const { user: otherUser } = await createTestUser();

      const res = await request(app)
        .put(`/api/auth/users/${otherUser._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          firstName: 'Hacked'
        })
        .expect(403);

      expect(res.body.message).toBe('Insufficient permissions');
    });

    it('should prevent last admin from demoting themselves', async () => {
      const { user, token } = await createAdminUser();

      const res = await request(app)
        .put(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          role: 'assessor'
        })
        .expect(400);

      expect(res.body.message).toBe('Cannot remove the last admin');
    });

    it('should allow admin demotion when other admins exist', async () => {
      const { user: admin1, token: admin1Token } = await createAdminUser();
      await createAdminUser(); // Second admin

      const res = await request(app)
        .put(`/api/auth/users/${admin1._id}`)
        .set('Authorization', `Bearer ${admin1Token}`)
        .send({
          role: 'assessor'
        })
        .expect(200);

      expect(res.body.user.role).toBe('assessor');
    });

    it('should return 404 for non-existent user', async () => {
      const { token } = await createAdminUser();
      const fakeId = new mongoose.Types.ObjectId();

      const res = await request(app)
        .put(`/api/auth/users/${fakeId}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(404);

      expect(res.body.message).toBe('User not found');
    });
  });

  describe('DELETE /api/auth/users/:id', () => {
    it('should return 401 without authentication', async () => {
      const { user } = await createTestUser();

      await request(app)
        .delete(`/api/auth/users/${user._id}`)
        .expect(401);
    });

    it('should return 403 for non-admin users', async () => {
      const { token } = await createTestUser();
      const { user: otherUser } = await createTestUser();

      await request(app)
        .delete(`/api/auth/users/${otherUser._id}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(403);
    });

    it('should delete user for admin', async () => {
      const { token } = await createAdminUser();
      const { user } = await createTestUser();

      const res = await request(app)
        .delete(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(res.body.message).toBe('User deleted successfully');

      // Verify user is deleted
      const deletedUser = await User.findById(user._id);
      expect(deletedUser).toBeNull();
    });

    it('should cascade delete user assessments', async () => {
      const { token } = await createAdminUser();
      const { user } = await createTestUser();

      // Create assessment for user
      await Assessment.create({
        userId: user._id,
        employeeName: 'Test Employee',
        metrics: {
          sharedVision: 3, strategy: 3, businessAlignment: 3, customerFocus: 3,
          crossFunctionalTeams: 3, clarityInPriorities: 3, acceptanceCriteria: 3,
          enablingFocus: 3, engagement: 3, feedback: 3, enablingAutonomy: 3,
          changeAndAmbiguity: 3, desiredCulture: 3, workAutonomously: 3,
          stakeholders: 3, teamAttrition: 3, teams: 3, developingPeople: 3,
          subordinatesForSuccess: 3
        }
      });

      // Verify assessment exists
      let assessments = await Assessment.find({ userId: user._id });
      expect(assessments).toHaveLength(1);

      // Delete user
      await request(app)
        .delete(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      // Verify assessments are deleted
      assessments = await Assessment.find({ userId: user._id });
      expect(assessments).toHaveLength(0);
    });

    it('should prevent self-deletion', async () => {
      const { user, token } = await createAdminUser();

      const res = await request(app)
        .delete(`/api/auth/users/${user._id}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(400);

      expect(res.body.message).toBe('Cannot delete your own account');
    });

    it('should allow deleting an admin when other admins remain', async () => {
      const { token } = await createAdminUser();
      const { user: otherAdmin } = await createAdminUser();

      await request(app)
        .delete(`/api/auth/users/${otherAdmin._id}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      const admins = await User.find({ role: 'admin' });
      expect(admins).toHaveLength(1);
    });

    // The "Cannot delete the last admin" guard in the controller can only fire when
    // requestingUser !== userToDelete and adminCount <= 1 at the same time. Via the API
    // that's unreachable: self-deletion is blocked earlier, and if the requesting admin
    // is themselves an admin, deleting a *different* admin means adminCount was >= 2
    // going in. So this is exercised as a controller-level unit test instead of a route test.
    it('should block deletion at the controller level when only one admin remains', async () => {
      const { deleteUser } = require('../controllers/authController');
      const { user: admin } = await createAdminUser();
      const { user: otherAdmin } = await createAdminUser();

      const countSpy = jest.spyOn(User, 'countDocuments').mockResolvedValue(1);

      const req = { params: { id: otherAdmin._id.toString() }, user: admin };
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };

      await deleteUser(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ message: 'Cannot delete the last admin' });

      countSpy.mockRestore();
      const stillExists = await User.findById(otherAdmin._id);
      expect(stillExists).not.toBeNull();
    });

    it('should return 404 for non-existent user', async () => {
      const { token } = await createAdminUser();
      const fakeId = new mongoose.Types.ObjectId();

      const res = await request(app)
        .delete(`/api/auth/users/${fakeId}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(404);

      expect(res.body.message).toBe('User not found');
    });
  });
});
