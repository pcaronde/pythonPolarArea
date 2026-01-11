/**
 * API Client for Performance Assessment Application
 * Handles all HTTP requests to the backend API
 */
class APIClient {
  constructor() {
    // Use dynamic URL based on current location
    this.baseURL = window.location.hostname === 'localhost'
      ? 'http://localhost:5000/api'
      : window.location.origin + '/api';

    this.token = localStorage.getItem('authToken');
  }

  /**
   * Generic request handler
   * @param {string} endpoint - API endpoint (e.g., '/auth/login')
   * @param {Object} options - Fetch options
   * @returns {Promise<Object>} - Response data
   */
  async request(endpoint, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    };

    // Add authorization token if available
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        ...options,
        headers
      });

      // Handle non-JSON responses (like CSV downloads)
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('text/csv')) {
        return response;
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // ==================== Authentication Methods ====================

  /**
   * Get authentication configuration
   * @returns {Promise<Object>} - { allowRegistration }
   */
  async getAuthConfig() {
    return this.request('/auth/config');
  }

  /**
   * Register a new user
   * @param {Object} userData - { email, password, firstName, lastName }
   * @returns {Promise<Object>} - { token, user }
   */
  async register(userData) {
    const data = await this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData)
    });

    // Store token
    this.token = data.token;
    localStorage.setItem('authToken', data.token);

    return data;
  }

  /**
   * Login with email and password
   * @param {string} email
   * @param {string} password
   * @returns {Promise<Object>} - { token, user }
   */
  async login(email, password) {
    const data = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password })
    });

    // Store token
    this.token = data.token;
    localStorage.setItem('authToken', data.token);

    return data;
  }

  /**
   * Logout user (remove token)
   * @returns {Promise<Object>}
   */
  async logout() {
    try {
      await this.request('/auth/logout', {
        method: 'POST'
      });
    } finally {
      // Clear token regardless of API response
      this.token = null;
      localStorage.removeItem('authToken');
      localStorage.removeItem('user');
      localStorage.removeItem('performanceAssessment');
    }
  }

  /**
   * Get current user info
   * @returns {Promise<Object>} - { user }
   */
  async getCurrentUser() {
    return this.request('/auth/me');
  }

  // ==================== Assessment Methods ====================

  /**
   * Get all assessments for logged-in user
   * @param {Object} filters - { page, limit, employeeName, startDate, endDate }
   * @returns {Promise<Object>} - { assessments, pagination }
   */
  async getAssessments(filters = {}) {
    const params = new URLSearchParams(filters).toString();
    const endpoint = params ? `/assessments?${params}` : '/assessments';
    return this.request(endpoint);
  }

  /**
   * Get single assessment by ID
   * @param {string} id - Assessment ID
   * @returns {Promise<Object>} - Assessment object
   */
  async getAssessment(id) {
    return this.request(`/assessments/${id}`);
  }

  /**
   * Create new assessment
   * @param {Object} data - { employeeName, assessmentDate?, metrics }
   * @returns {Promise<Object>} - { assessment }
   */
  async createAssessment(data) {
    return this.request('/assessments', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  /**
   * Update existing assessment
   * @param {string} id - Assessment ID
   * @param {Object} data - { employeeName?, assessmentDate?, metrics? }
   * @returns {Promise<Object>} - { assessment }
   */
  async updateAssessment(id, data) {
    return this.request(`/assessments/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  /**
   * Delete assessment
   * @param {string} id - Assessment ID
   * @returns {Promise<Object>} - { message }
   */
  async deleteAssessment(id) {
    return this.request(`/assessments/${id}`, {
      method: 'DELETE'
    });
  }

  /**
   * Import assessment from CSV file
   * @param {File} file - CSV file
   * @param {string} employeeName - Employee name
   * @returns {Promise<Object>} - { assessment, errors? }
   */
  async importFromCSV(file, employeeName) {
    const formData = new FormData();
    formData.append('csvFile', file);
    formData.append('employeeName', employeeName);

    // Don't set Content-Type header, let browser set it with boundary
    const headers = {};
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    const response = await fetch(`${this.baseURL}/assessments/import-csv`, {
      method: 'POST',
      headers,
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Import failed');
    }

    return data;
  }

  /**
   * Export assessments to CSV
   * @param {Object} filters - { ids?, employeeName?, startDate?, endDate? }
   * @returns {Promise<Blob>} - CSV file blob
   */
  async exportToCSV(filters = {}) {
    const params = new URLSearchParams(filters).toString();
    const endpoint = params ? `/assessments/export-csv?${params}` : '/assessments/export-csv';

    const response = await this.request(endpoint);

    // Response is already the fetch Response object for CSV
    const blob = await response.blob();
    return blob;
  }
}

// Create singleton instance
const api = new APIClient();

// Export for use in other modules
window.api = api;
