module.exports = {
  testEnvironment: 'node',
  setupFilesAfterEnv: ['<rootDir>/backend/__tests__/setup.js'],
  testMatch: ['**/__tests__/**/*.test.js'],
  testPathIgnorePatterns: ['/node_modules/'],
  collectCoverageFrom: [
    'backend/**/*.js',
    '!backend/__tests__/**',
    '!backend/server.js'
  ],
  coverageDirectory: 'coverage',
  verbose: true,
  testTimeout: 30000
};
