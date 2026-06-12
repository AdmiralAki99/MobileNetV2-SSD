import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Header, type ViewMode } from '../../src/components/Header'

const defaultProps = {
  viewMode: 'pipeline' as ViewMode,
  setViewMode: jest.fn(),
  statusFilter: 'all',
  setStatusFilter: jest.fn(),
}

describe('Header', () => {
  beforeEach(() => jest.clearAllMocks())

  test('renders the sentinel brand', () => {
    render(<Header {...defaultProps} />)
    expect(screen.getByText('sentinel')).toBeInTheDocument()
  })

  test('renders all six tabs', () => {
    render(<Header {...defaultProps} />)
    ;['Pipeline', 'Metrics', 'ETL', 'Ops', 'Deploy', 'Config'].forEach(t =>
      expect(screen.getByText(t)).toBeInTheDocument()
    )
  })

  test('shows status filter pills in pipeline view', () => {
    render(<Header {...defaultProps} viewMode="pipeline" />)
    ;['all', 'pending', 'running', 'success', 'failed'].forEach(s =>
      expect(screen.getByText(s)).toBeInTheDocument()
    )
  })

  test('hides status filter pills in non-pipeline views', () => {
    render(<Header {...defaultProps} viewMode="metrics" />)
    expect(screen.queryByText('pending')).not.toBeInTheDocument()
  })

  test('calls setViewMode when a tab is clicked', async () => {
    const setViewMode = jest.fn()
    render(<Header {...defaultProps} setViewMode={setViewMode} />)
    await userEvent.click(screen.getByText('Metrics'))
    expect(setViewMode).toHaveBeenCalledWith('metrics')
  })

  test('calls setStatusFilter when a filter pill is clicked', async () => {
    const setStatusFilter = jest.fn()
    render(<Header {...defaultProps} setStatusFilter={setStatusFilter} />)
    await userEvent.click(screen.getByText('running'))
    expect(setStatusFilter).toHaveBeenCalledWith('running')
  })

  test('active status pill matches statusFilter prop', () => {
    render(<Header {...defaultProps} statusFilter="running" />)
    const btn = screen.getByText('running')
    expect(btn.style.background).toBe('rgb(255, 255, 255)')
  })

  test('renders the search bar', () => {
    render(<Header {...defaultProps} />)
    expect(screen.getByPlaceholderText('Search by Name etc')).toBeInTheDocument()
  })

  test('renders the gear button', () => {
    render(<Header {...defaultProps} />)
    const buttons = screen.getAllByRole('button')
    const gearBtn = buttons.find(b => b.querySelector('svg') && !b.textContent?.trim())
    expect(gearBtn).toBeInTheDocument()
  })
})
