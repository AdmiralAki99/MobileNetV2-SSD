import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../src/App'

describe('App', () => {
  test('renders the header', () => {
    render(<App />)
    expect(screen.getByText('sentinel')).toBeInTheDocument()
  })

  test('defaults to pipeline view', () => {
    render(<App />)
    expect(screen.getByTestId('view-pipeline')).toBeInTheDocument()
  })

  test('respects initialView prop', () => {
    render(<App initialView="metrics" />)
    expect(screen.getByTestId('view-metrics')).toBeInTheDocument()
  })

  test('switching tabs changes the active view', async () => {
    render(<App />)
    await userEvent.click(screen.getByText('ETL'))
    expect(screen.getByTestId('view-etl')).toBeInTheDocument()
    expect(screen.queryByTestId('view-pipeline')).not.toBeInTheDocument()
  })

  test('status filter pills only visible in pipeline view', async () => {
    render(<App />)
    expect(screen.getByText('pending')).toBeInTheDocument()
    await userEvent.click(screen.getByText('Metrics'))
    expect(screen.queryByText('pending')).not.toBeInTheDocument()
  })

  test('status filter persists when switching back to pipeline', async () => {
    render(<App />)
    await userEvent.click(screen.getByText('running'))
    await userEvent.click(screen.getByText('Metrics'))
    await userEvent.click(screen.getByText('Pipeline'))
    const runningBtn = screen.getByText('running')
    expect(runningBtn.style.background).toBe('rgb(255, 255, 255)')
  })

  test('all six views can be navigated to', async () => {
    render(<App />)
    const views: Array<[string, string]> = [
      ['Metrics', 'view-metrics'], ['ETL', 'view-etl'], ['Ops', 'view-ops'],
      ['Deploy', 'view-deploy'], ['Config', 'view-config'], ['Pipeline', 'view-pipeline'],
    ]
    for (const [tab, testId] of views) {
      await userEvent.click(screen.getByText(tab))
      expect(screen.getByTestId(testId)).toBeInTheDocument()
    }
  })
})
