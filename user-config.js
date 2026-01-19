/**
 * User Configuration Page Script
 * Handles user profile updates and admin user management
 */

// Authentication guard
if (!authManager.isAuthenticated()) {
    window.location.href = '/login.html?returnUrl=' + encodeURIComponent(window.location.pathname);
}

// Current user data
let currentUser = authManager.getUser();
let allUsers = [];
let userToDelete = null;

// Display logged-in user
if (currentUser) {
    document.getElementById('userDisplay').textContent = `${currentUser.firstName} ${currentUser.lastName}`;
}

/**
 * Logout handler
 */
function handleLogout() {
    if (confirm('Are you sure you want to logout?')) {
        authManager.logout();
    }
}

/**
 * Show message to user
 */
function showMessage(message, type = 'error') {
    const container = document.getElementById('messageContainer');
    const messageClass = type === 'error' ? 'error-message' : 'success-message';
    container.innerHTML = `<div class="${messageClass}">${message}</div>`;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        container.innerHTML = '';
    }, 5000);

    // Scroll to message
    container.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Load current user profile into form
 */
function loadUserProfile() {
    if (!currentUser) return;

    document.getElementById('profileFirstName').value = currentUser.firstName || '';
    document.getElementById('profileLastName').value = currentUser.lastName || '';
    document.getElementById('profileEmail').value = currentUser.email || '';
}

/**
 * Handle profile form submission
 */
document.getElementById('profileForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const firstName = document.getElementById('profileFirstName').value.trim();
    const lastName = document.getElementById('profileLastName').value.trim();
    const email = document.getElementById('profileEmail').value.trim();
    const currentPassword = document.getElementById('currentPassword').value;
    const newPassword = document.getElementById('newPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    // Validate password confirmation
    if (newPassword && newPassword !== confirmPassword) {
        showMessage('New passwords do not match');
        return;
    }

    // Validate password length
    if (newPassword && newPassword.length < 6) {
        showMessage('Password must be at least 6 characters');
        return;
    }

    // Require current password if changing password or email
    const emailChanged = email !== currentUser.email;
    if ((newPassword || emailChanged) && !currentPassword) {
        showMessage('Current password is required to change email or password');
        return;
    }

    const saveBtn = document.getElementById('saveProfileBtn');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    try {
        const updateData = { firstName, lastName, email };

        if (currentPassword) {
            updateData.currentPassword = currentPassword;
        }

        if (newPassword) {
            updateData.password = newPassword;
        }

        const response = await api.updateProfile(updateData);

        // Update local user data
        currentUser = response.user;
        localStorage.setItem('user', JSON.stringify(response.user));
        authManager.user = response.user;

        // Update display
        document.getElementById('userDisplay').textContent = `${currentUser.firstName} ${currentUser.lastName}`;

        // Clear password fields
        document.getElementById('currentPassword').value = '';
        document.getElementById('newPassword').value = '';
        document.getElementById('confirmPassword').value = '';

        showMessage('Profile updated successfully', 'success');
    } catch (error) {
        showMessage(error.message || 'Failed to update profile');
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save Changes';
    }
});

/**
 * Load all users (admin only)
 */
async function loadAllUsers() {
    try {
        const response = await api.getAllUsers();
        allUsers = response.users;
        renderUsersTable();
    } catch (error) {
        console.error('Failed to load users:', error);
        document.getElementById('usersTableBody').innerHTML =
            '<tr><td colspan="5" class="error-cell">Failed to load users</td></tr>';
    }
}

/**
 * Render users table
 */
function renderUsersTable() {
    const tbody = document.getElementById('usersTableBody');

    if (!allUsers || allUsers.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No users found</td></tr>';
        return;
    }

    tbody.innerHTML = allUsers.map(user => {
        const isSelf = user._id === currentUser.id || user.id === currentUser.id;
        const createdDate = new Date(user.createdAt).toLocaleDateString();

        return `
            <tr class="${isSelf ? 'current-user-row' : ''}">
                <td>${escapeHtml(user.firstName)} ${escapeHtml(user.lastName)}${isSelf ? ' (you)' : ''}</td>
                <td>${escapeHtml(user.email)}</td>
                <td><span class="role-badge role-${user.role}">${user.role}</span></td>
                <td>${createdDate}</td>
                <td class="actions-cell">
                    <button onclick="editUser('${user._id || user.id}')" class="btn-small btn-edit">Edit</button>
                    ${!isSelf ? `<button onclick="showDeleteModal('${user._id || user.id}')" class="btn-small btn-delete">Delete</button>` : ''}
                </td>
            </tr>
        `;
    }).join('');
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show add user modal
 */
function showAddUserModal() {
    document.getElementById('modalTitle').textContent = 'Add New User';
    document.getElementById('editUserId').value = '';
    document.getElementById('userForm').reset();
    document.getElementById('userPassword').required = true;
    document.getElementById('passwordHint').textContent = '*';
    document.getElementById('passwordEditHint').style.display = 'none';
    document.getElementById('saveUserBtn').textContent = 'Add User';
    document.getElementById('userModal').classList.remove('hidden');
}

/**
 * Edit user
 */
function editUser(userId) {
    const user = allUsers.find(u => (u._id || u.id) === userId);
    if (!user) return;

    document.getElementById('modalTitle').textContent = 'Edit User';
    document.getElementById('editUserId').value = userId;
    document.getElementById('userFirstName').value = user.firstName;
    document.getElementById('userLastName').value = user.lastName;
    document.getElementById('userEmail').value = user.email;
    document.getElementById('userPassword').value = '';
    document.getElementById('userPassword').required = false;
    document.getElementById('passwordHint').textContent = '';
    document.getElementById('passwordEditHint').style.display = 'block';
    document.getElementById('userRole').value = user.role;
    document.getElementById('saveUserBtn').textContent = 'Save Changes';
    document.getElementById('userModal').classList.remove('hidden');
}

/**
 * Close user modal
 */
function closeUserModal() {
    document.getElementById('userModal').classList.add('hidden');
}

/**
 * Handle user form submission
 */
document.getElementById('userForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const userId = document.getElementById('editUserId').value;
    const isEdit = !!userId;

    const userData = {
        firstName: document.getElementById('userFirstName').value.trim(),
        lastName: document.getElementById('userLastName').value.trim(),
        email: document.getElementById('userEmail').value.trim(),
        role: document.getElementById('userRole').value
    };

    const password = document.getElementById('userPassword').value;
    if (password) {
        userData.password = password;
    } else if (!isEdit) {
        showMessage('Password is required for new users');
        return;
    }

    const saveBtn = document.getElementById('saveUserBtn');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    try {
        if (isEdit) {
            await api.updateUser(userId, userData);
            showMessage('User updated successfully', 'success');
        } else {
            await api.createUser(userData);
            showMessage('User created successfully', 'success');
        }

        closeUserModal();
        loadAllUsers();
    } catch (error) {
        showMessage(error.message || 'Failed to save user');
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = isEdit ? 'Save Changes' : 'Add User';
    }
});

/**
 * Show delete confirmation modal
 */
function showDeleteModal(userId) {
    const user = allUsers.find(u => (u._id || u.id) === userId);
    if (!user) return;

    userToDelete = userId;
    document.getElementById('deleteMessage').textContent =
        `Are you sure you want to delete ${user.firstName} ${user.lastName} (${user.email})? This will also delete all their assessments.`;
    document.getElementById('deleteModal').classList.remove('hidden');
}

/**
 * Close delete modal
 */
function closeDeleteModal() {
    document.getElementById('deleteModal').classList.add('hidden');
    userToDelete = null;
}

/**
 * Confirm delete user
 */
async function confirmDeleteUser() {
    if (!userToDelete) return;

    const deleteBtn = document.getElementById('confirmDeleteBtn');
    deleteBtn.disabled = true;
    deleteBtn.textContent = 'Deleting...';

    try {
        await api.deleteUser(userToDelete);
        showMessage('User deleted successfully', 'success');
        closeDeleteModal();
        loadAllUsers();
    } catch (error) {
        showMessage(error.message || 'Failed to delete user');
    } finally {
        deleteBtn.disabled = false;
        deleteBtn.textContent = 'Delete';
    }
}

/**
 * Initialize page
 */
function initPage() {
    // Load user profile
    loadUserProfile();

    // Show admin section if user is admin
    if (currentUser && currentUser.role === 'admin') {
        document.getElementById('adminSection').classList.remove('hidden');
        loadAllUsers();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initPage);

// Close modals on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeUserModal();
        closeDeleteModal();
    }
});

// Close modals when clicking outside
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeUserModal();
            closeDeleteModal();
        }
    });
});
