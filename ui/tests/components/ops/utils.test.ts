import { fmtDur, fmtTime } from '../../../src/components/ops/utils'

describe('fmtDur', () => {
  test('returns — for null', () => expect(fmtDur(null)).toBe('—'))
  test('returns — for undefined', () => expect(fmtDur(undefined)).toBe('—'))
  test('formats seconds under 60', () => expect(fmtDur(12.3)).toBe('12.3s'))
  test('formats exactly 60s as minutes', () => expect(fmtDur(60)).toBe('1.0m'))
  test('formats minutes', () => expect(fmtDur(483.2)).toBe('8.1m'))
  test('formats zero', () => expect(fmtDur(0)).toBe('0.0s'))
})

describe('fmtTime', () => {
  test('returns — for null', () => expect(fmtTime(null)).toBe('—'))
  test('returns — for undefined', () => expect(fmtTime(undefined)).toBe('—'))
  test('returns a time string for a valid ISO date', () => {
    const result = fmtTime('2026-05-23T02:00:02Z')
    expect(result).toMatch(/\d{2}:\d{2}:\d{2}/)
  })
})
