import {
  fetchExperiments, fetchInstanceStatus, launchTraining,
} from '../../src/api/client'

describe('API client', () => {
  beforeEach(() => {
    global.fetch = jest.fn()
  })
  afterEach(() => jest.resetAllMocks())

  test('fetchExperiments calls correct endpoint', async () => {
    (fetch as jest.Mock).mockResolvedValue({
      ok: true, json: async () => [],
    })
    await fetchExperiments()
    expect(fetch).toHaveBeenCalledWith('/api/experiments', {})
  })

  test('fetchInstanceStatus calls endpoint with id', async () => {
    (fetch as jest.Mock).mockResolvedValue({
      ok: true, json: async () => ({}),
    })
    await fetchInstanceStatus('i-abc123')
    expect(fetch).toHaveBeenCalledWith('/api/training/i-abc123/status', {})
  })

  test('launchTraining posts JSON body', async () => {
    (fetch as jest.Mock).mockResolvedValue({
      ok: true, json: async () => ({ status: 'ok' }),
    })
    await launchTraining({ experiment_id: 'exp003' })
    expect(fetch).toHaveBeenCalledWith('/api/training/launch', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ experiment_id: 'exp003' }),
    }))
  })

  test('throws on non-ok response', async () => {
    (fetch as jest.Mock).mockResolvedValue({
      ok: false, status: 500, text: async () => 'Internal Server Error',
    })
    await expect(fetchExperiments()).rejects.toThrow('500: Internal Server Error')
  })
})
