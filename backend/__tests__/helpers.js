/**
 * Test Helpers
 * Utility functions for tests
 */

const User = require('../models/User');
const { generateToken } = require('../config/jwt');

/**
 * Create a test user and return user object with token
 */
async function createTestUser(overrides = {}) {
  const userData = {
    email: `test-${Date.now()}@example.com`,
    password: 'password123',
    firstName: 'Test',
    lastName: 'User',
    role: 'assessor',
    ...overrides
  };

  const user = new User(userData);
  await user.save();

  const token = generateToken({
    userId: user._id,
    email: user.email,
    role: user.role
  });

  return { user, token };
}

/**
 * Create an admin user
 */
async function createAdminUser(overrides = {}) {
  return createTestUser({ role: 'admin', ...overrides });
}

/**
 * Create multiple test users
 */
async function createTestUsers(count, overrides = {}) {
  const users = [];
  for (let i = 0; i < count; i++) {
    const { user, token } = await createTestUser({
      email: `test-user-${i}-${Date.now()}@example.com`,
      firstName: `User${i}`,
      ...overrides
    });
    users.push({ user, token });
  }
  return users;
}

module.exports = {
  createTestUser,
  createAdminUser,
  createTestUsers
};
