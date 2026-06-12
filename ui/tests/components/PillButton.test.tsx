import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PillButton } from '../../src/components/PillButton'

describe('PillButton', () => {
  test('renders children', () => {
    render(<PillButton>click me</PillButton>)
    expect(screen.getByText('click me')).toBeInTheDocument()
  })

  test('calls onClick when clicked', async () => {
    const handler = jest.fn()
    render(<PillButton onClick={handler}>click me</PillButton>)
    await userEvent.click(screen.getByText('click me'))
    expect(handler).toHaveBeenCalledTimes(1)
  })

  test('active state applies white background', () => {
    const { container } = render(<PillButton active>active</PillButton>)
    const btn = container.querySelector('button')!
    expect(btn.style.background).toBe('rgb(255, 255, 255)')
  })

  test('inactive state applies pill background', () => {
    const { container } = render(<PillButton>inactive</PillButton>)
    const btn = container.querySelector('button')!
    expect(btn.style.background).toBe('var(--bg-pill)')
  })

  test('accepts and merges custom style', () => {
    const { container } = render(<PillButton style={{ opacity: 0.5 }}>styled</PillButton>)
    const btn = container.querySelector('button')!
    expect(btn.style.opacity).toBe('0.5')
  })

  test('does not throw when onClick is omitted', async () => {
    render(<PillButton>no handler</PillButton>)
    await userEvent.click(screen.getByText('no handler'))
  })
})
