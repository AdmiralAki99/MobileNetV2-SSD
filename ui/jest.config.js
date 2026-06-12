export default {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['@testing-library/jest-dom', '<rootDir>/tests/setupTests.ts'],
  transform: {
    '^.+\\.[jt]sx?$': 'babel-jest',
    '^.+\\.js$': 'babel-jest',
  },
  transformIgnorePatterns: [
    '/node_modules/(?!(d3|d3-.*|internmap|delaunator|robust-predicates|topojson-client)/)',
  ],
  moduleNameMapper: {
    '\\.(css|svg)$': '<rootDir>/tests/__mocks__/fileMock.js',
  },
  testMatch: ['**/tests/**/*.test.[jt]s?(x)'],
}
