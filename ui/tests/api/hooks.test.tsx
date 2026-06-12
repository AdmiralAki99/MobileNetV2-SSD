import { renderHook, act } from '@testing-library/react'
import { usePolling } from '../../src/api/hooks'

describe('usePolling', () => {
  beforeEach(() => jest.useFakeTimers())
  afterEach(() => jest.useRealTimers())

  test('starts with null data and loading false', () => {
    const fn = jest.fn().mockResolvedValue({ ok: true })
    const { result } = renderHook(() => usePolling(fn, 5000, false))
    expect(result.current.data).toBeNull()
    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  test('calls fn immediately when enabled', async () => {
    const fn = jest.fn().mockResolvedValue({ ok: true })
    renderHook(() => usePolling(fn, 5000, true))
    await act(async () => {})
    expect(fn).toHaveBeenCalledTimes(1)
  })

  test('does not call fn when disabled', async () => {
    const fn = jest.fn().mockResolvedValue({ ok: true })
    renderHook(() => usePolling(fn, 5000, false))
    await act(async () => {})
    expect(fn).toHaveBeenCalledTimes(0)
  })

  test('sets data on success', async () => {
    const fn = jest.fn().mockResolvedValue({ value: 42 })
    const { result } = renderHook(() => usePolling(fn, 5000, true))
    await act(async () => {})
    expect(result.current.data).toEqual({ value: 42 })
    expect(result.current.error).toBeNull()
  })

  test('sets error on failure', async () => {
    const fn = jest.fn().mockRejectedValue(new Error('network error'))
    const { result } = renderHook(() => usePolling(fn, 5000, true))
    await act(async () => {})
    expect(result.current.error).toBe('network error')
    expect(result.current.data).toBeNull()
  })

  test('polls again after interval', async () => {
    const fn = jest.fn().mockResolvedValue({})
    renderHook(() => usePolling(fn, 5000, true))
    await act(async () => {})
    expect(fn).toHaveBeenCalledTimes(1)
    await act(async () => { jest.advanceTimersByTime(5000) })
    expect(fn).toHaveBeenCalledTimes(2)
  })

  test('refresh triggers an immediate call', async () => {
    const fn = jest.fn().mockResolvedValue({})
    const { result } = renderHook(() => usePolling(fn, 5000, true))
    await act(async () => {})
    expect(fn).toHaveBeenCalledTimes(1)
    await act(async () => { result.current.refresh() })
    expect(fn).toHaveBeenCalledTimes(2)
  })
})
