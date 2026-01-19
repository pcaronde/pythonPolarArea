const User = require('../models/User');
const Assessment = require('../models/Assessment');
const { generateToken } = require('../config/jwt');

/**
 * Register a new user
 * POST /api/auth/register
 */
async function register(req, res) {
  try {
    // Check if registration is allowed
    const allowRegistration = process.env.ALLOW_REGISTRATION === 'true';
    if (!allowRegistration) {
      return res.status(403).json({
        message: 'Registration is currently disabled. Please contact an administrator.'
      });
    }

    const { email, password, firstName, lastName } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ message: 'Email already registered' });
    }

    // Create new user
    const user = new User({
      email,
      password, // Will be hashed by pre-save hook
      firstName,
      lastName
    });

    await user.save();

    // Generate JWT token
    const token = generateToken({
      userId: user._id,
      email: user.email,
      role: user.role
    });

    // Return token and user info
    res.status(201).json({
      message: 'User registered successfully',
      token,
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role
      }
    });

  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ message: 'Failed to register user' });
  }
}

/**
 * Login user
 * POST /api/auth/login
 */
async function login(req, res) {
  try {
    const { email, password } = req.body;

    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Check password
    const isPasswordValid = await user.comparePassword(password);
    if (!isPasswordValid) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Update last login time
    user.lastLogin = new Date();
    await user.save();

    // Generate JWT token
    const token = generateToken({
      userId: user._id,
      email: user.email,
      role: user.role
    });

    // Return token and user info
    res.json({
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        lastLogin: user.lastLogin
      }
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Failed to login' });
  }
}

/**
 * Get current user info
 * GET /api/auth/me
 */
async function getCurrentUser(req, res) {
  try {
    // req.user is populated by authMiddleware
    res.json({
      user: {
        id: req.user._id,
        email: req.user.email,
        firstName: req.user.firstName,
        lastName: req.user.lastName,
        role: req.user.role,
        createdAt: req.user.createdAt,
        lastLogin: req.user.lastLogin
      }
    });
  } catch (error) {
    console.error('Get current user error:', error);
    res.status(500).json({ message: 'Failed to get user info' });
  }
}

/**
 * Get registration configuration
 * GET /api/auth/config
 */
async function getAuthConfig(req, res) {
  try {
    const allowRegistration = process.env.ALLOW_REGISTRATION === 'true';
    res.json({
      allowRegistration
    });
  } catch (error) {
    console.error('Get auth config error:', error);
    res.status(500).json({ message: 'Failed to get configuration' });
  }
}

/**
 * Logout user (client-side token removal)
 * POST /api/auth/logout
 */
function logout(req, res) {
  // With JWT, logout is handled client-side by removing the token
  // This endpoint is here for consistency and future enhancements (blacklist, etc.)
  res.json({ message: 'Logged out successfully' });
}

/**
 * Get all users (admin only)
 * GET /api/auth/users
 */
async function getAllUsers(req, res) {
  try {
    const users = await User.find().select('-password').sort({ createdAt: -1 });
    res.json({ users });
  } catch (error) {
    console.error('Get all users error:', error);
    res.status(500).json({ message: 'Failed to get users' });
  }
}

/**
 * Create a new user (admin only)
 * POST /api/auth/users
 */
async function createUser(req, res) {
  try {
    const { email, password, firstName, lastName, role } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ message: 'Email already registered' });
    }

    // Create new user
    const user = new User({
      email,
      password, // Will be hashed by pre-save hook
      firstName,
      lastName,
      role: role || 'assessor'
    });

    await user.save();

    // Return user info (without password)
    res.status(201).json({
      message: 'User created successfully',
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    console.error('Create user error:', error);
    res.status(500).json({ message: 'Failed to create user' });
  }
}

/**
 * Update user (self or admin)
 * PUT /api/auth/users/:id
 */
async function updateUser(req, res) {
  try {
    const { id } = req.params;
    const { email, password, firstName, lastName, role, currentPassword } = req.body;
    const requestingUser = req.user;

    // Check authorization: user can edit self, admin can edit anyone
    const isSelf = requestingUser._id.toString() === id;
    const isAdmin = requestingUser.role === 'admin';

    if (!isSelf && !isAdmin) {
      return res.status(403).json({ message: 'Insufficient permissions' });
    }

    // Find the user to update
    const userToUpdate = await User.findById(id);
    if (!userToUpdate) {
      return res.status(404).json({ message: 'User not found' });
    }

    // For self-update, require current password if changing password or email
    const emailChanging = email && email !== userToUpdate.email;
    if (isSelf && !isAdmin && (password || emailChanging)) {
      if (!currentPassword) {
        return res.status(400).json({ message: 'Current password required for this change' });
      }
      const isPasswordValid = await userToUpdate.comparePassword(currentPassword);
      if (!isPasswordValid) {
        return res.status(401).json({ message: 'Current password is incorrect' });
      }
    }

    // Prevent changing role for non-admins
    if (role && !isAdmin) {
      return res.status(403).json({ message: 'Only admins can change user roles' });
    }

    // Prevent admin from demoting themselves if they're the last admin
    if (isAdmin && isSelf && role && role !== 'admin') {
      const adminCount = await User.countDocuments({ role: 'admin' });
      if (adminCount <= 1) {
        return res.status(400).json({ message: 'Cannot remove the last admin' });
      }
    }

    // Check for email uniqueness if changing email
    if (email && email !== userToUpdate.email) {
      const emailExists = await User.findOne({ email, _id: { $ne: id } });
      if (emailExists) {
        return res.status(409).json({ message: 'Email already in use' });
      }
      userToUpdate.email = email;
    }

    // Update fields
    if (firstName) userToUpdate.firstName = firstName;
    if (lastName) userToUpdate.lastName = lastName;
    if (password) userToUpdate.password = password; // Will be hashed by pre-save hook
    if (role && isAdmin) userToUpdate.role = role;

    await userToUpdate.save();

    // Return updated user info
    res.json({
      message: 'User updated successfully',
      user: {
        id: userToUpdate._id,
        email: userToUpdate.email,
        firstName: userToUpdate.firstName,
        lastName: userToUpdate.lastName,
        role: userToUpdate.role,
        createdAt: userToUpdate.createdAt,
        lastLogin: userToUpdate.lastLogin
      }
    });
  } catch (error) {
    console.error('Update user error:', error);
    res.status(500).json({ message: 'Failed to update user' });
  }
}

/**
 * Delete user (admin only)
 * DELETE /api/auth/users/:id
 */
async function deleteUser(req, res) {
  try {
    const { id } = req.params;
    const requestingUser = req.user;

    // Prevent self-deletion
    if (requestingUser._id.toString() === id) {
      return res.status(400).json({ message: 'Cannot delete your own account' });
    }

    // Find the user to delete
    const userToDelete = await User.findById(id);
    if (!userToDelete) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Prevent deleting the last admin
    if (userToDelete.role === 'admin') {
      const adminCount = await User.countDocuments({ role: 'admin' });
      if (adminCount <= 1) {
        return res.status(400).json({ message: 'Cannot delete the last admin' });
      }
    }

    // Delete user's assessments (cascade delete)
    await Assessment.deleteMany({ userId: id });

    // Delete the user
    await User.findByIdAndDelete(id);

    res.json({ message: 'User deleted successfully' });
  } catch (error) {
    console.error('Delete user error:', error);
    res.status(500).json({ message: 'Failed to delete user' });
  }
}

module.exports = {
  register,
  login,
  getCurrentUser,
  getAuthConfig,
  logout,
  getAllUsers,
  createUser,
  updateUser,
  deleteUser
};
