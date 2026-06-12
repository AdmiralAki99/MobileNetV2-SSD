import { render, screen } from '@testing-library/react'
import { StatusBadge } from '../../src/components/StatusBadge'
import { STATUS_COLOR, type Status } from '../../src/constants/status'

describe('StatusBadge', () => {
  test('renders the status text', () => {
    render(<StatusBadge status="running" />)
    expect(screen.getByText('running')).toBeInTheDocument()
  })

  test('renders "unknown" when no status given', () => {
    render(<StatusBadge />)
    expect(screen.getByText('unknown')).toBeInTheDocument()
  })

  test('renders all known statuses without crashing', () => {
    const statuses = Object.keys(STATUS_COLOR) as Status[]
    statuses.forEach(s => {
      const { unmount } = render(<StatusBadge status={s} />)
      expect(screen.getByText(s)).toBeInTheDocument()
      unmount()
    })
  })

  test('applies uppercase text transform', () => {
    const { container } = render(<StatusBadge status="pending" />)
    const span = container.querySelector('span')!
    expect(span.style.textTransform).toBe('uppercase')
  })

  test('falls back to #555 color for unknown status', () => {
    const { container } = render(<StatusBadge status="bogus" />)
    const span = container.querySelector('span')!
    expect(span.style.color).toBe('rgb(85, 85, 85)')
  })
})
