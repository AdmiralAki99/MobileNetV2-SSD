import { render, screen } from '@testing-library/react'
import { AlertBadge } from '../../src/components/AlertBadge'

describe('AlertBadge', () => {
  test('renders without crashing', () => {
    render(<AlertBadge />)
    expect(screen.getByRole('status')).toBeInTheDocument()
  })

  test('defaults to protected type', () => {
    render(<AlertBadge />)
    expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'protected')
  })

  test('renders danger type', () => {
    render(<AlertBadge type="danger" />)
    expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'danger')
  })

  test('renders alert type', () => {
    render(<AlertBadge type="alert" />)
    expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'alert')
  })

  test('animate=true sets pulse animation', () => {
    const { container } = render(<AlertBadge animate={true} />)
    const div = container.firstChild as HTMLElement
    expect(div.style.animation).toContain('pulse')
  })

  test('animate=false sets no animation', () => {
    const { container } = render(<AlertBadge animate={false} />)
    const div = container.firstChild as HTMLElement
    expect(div.style.animation).toBe('none')
  })

  test('accepts custom style', () => {
    const { container } = render(<AlertBadge style={{ width: 24, height: 24 }} />)
    const div = container.firstChild as HTMLElement
    expect(div.style.width).toBe('24px')
    expect(div.style.height).toBe('24px')
  })

  test('renders an svg icon inside', () => {
    const { container } = render(<AlertBadge />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })
})
