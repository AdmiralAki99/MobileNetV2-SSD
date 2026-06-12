import { useState, useEffect, useRef, useCallback } from 'react'

interface PollingResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refresh: () => void
}

export function usePolling<T>(
  fn: () => Promise<T>,
  interval = 5000,
  enabled = true,
): PollingResult<T> {
  const [data, setData]       = useState<T | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const fnRef = useRef(fn)
  fnRef.current = fn

  const call = useCallback(async () => {
    setLoading(true)
    try {
      const result = await fnRef.current()
      setData(result)
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!enabled) return
    call()
    const id = setInterval(call, interval)
    return () => clearInterval(id)
  }, [enabled, interval, call])

  return { data, loading, error, refresh: call }
}
