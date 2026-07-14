/**
 * User Model Tests
 * Tests for password hashing, comparison, and serialization
 */

const User = require('../models/User');

describe('User Model', () => {
  it('should hash the password on save', async () => {
    const user = new User({
      email: 'hash@example.com',
      password: 'plaintext123',
      firstName: 'Hash',
      lastName: 'Test'
    });

    await user.save();

    expect(user.password).not.toBe('plaintext123');
    expect(user.password.length).toBeGreaterThan('plaintext123'.length);
  });

  it('should not re-hash the password when unrelated fields are updated', async () => {
    const user = new User({
      email: 'rehash@example.com',
      password: 'plaintext123',
      firstName: 'Rehash',
      lastName: 'Test'
    });
    await user.save();
    const originalHash = user.password;

    user.firstName = 'Updated';
    await user.save();

    expect(user.password).toBe(originalHash);
  });

  it('should re-hash the password when it is modified', async () => {
    const user = new User({
      email: 'change@example.com',
      password: 'plaintext123',
      firstName: 'Change',
      lastName: 'Test'
    });
    await user.save();
    const originalHash = user.password;

    user.password = 'newplaintext456';
    await user.save();

    expect(user.password).not.toBe(originalHash);
  });

  describe('comparePassword', () => {
    it('should return true for the correct password', async () => {
      const user = new User({
        email: 'compare@example.com',
        password: 'correctpassword',
        firstName: 'Compare',
        lastName: 'Test'
      });
      await user.save();

      await expect(user.comparePassword('correctpassword')).resolves.toBe(true);
    });

    it('should return false for an incorrect password', async () => {
      const user = new User({
        email: 'compare2@example.com',
        password: 'correctpassword',
        firstName: 'Compare',
        lastName: 'Test'
      });
      await user.save();

      await expect(user.comparePassword('wrongpassword')).resolves.toBe(false);
    });
  });

  describe('toJSON', () => {
    it('should strip password and __v from the serialized output', async () => {
      const user = new User({
        email: 'serialize@example.com',
        password: 'plaintext123',
        firstName: 'Serialize',
        lastName: 'Test'
      });
      await user.save();

      const json = user.toJSON();

      expect(json).not.toHaveProperty('password');
      expect(json).not.toHaveProperty('__v');
      expect(json.email).toBe('serialize@example.com');
    });
  });

  describe('email uniqueness', () => {
    it('should reject a duplicate email at the database level', async () => {
      await new User({
        email: 'dup@example.com',
        password: 'plaintext123',
        firstName: 'Dup',
        lastName: 'One'
      }).save();

      const duplicate = new User({
        email: 'dup@example.com',
        password: 'plaintext123',
        firstName: 'Dup',
        lastName: 'Two'
      });

      await expect(duplicate.save()).rejects.toThrow();
    });
  });
});
