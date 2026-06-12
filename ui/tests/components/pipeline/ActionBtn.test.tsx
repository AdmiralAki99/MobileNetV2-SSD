import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ActionBtn } from '../../../src/components/pipeline/ActionBtn'

describe('ActionBtn', () => {
  test('renders the label', () => {
    render(<ActionBtn label="Launch" color="#65c16a" />)
    expect(screen.getByText('Launch')).toBeInTheDocument()
  })

  test('shows … when busy', () => {
    render(<ActionBtn label="Launch" color="#65c16a" busy />)
    expect(screen.getByText('…')).toBeInTheDocument()
    expect(screen.queryByText('Launch')).not.toBeInTheDocument()
  })

  test('calls onClick when enabled', async () => {
    const handler = jest.fn()
    render(<ActionBtn label="Launch" color="#65c16a" onClick={handler} />)
    await userEvent.click(screen.getByRole('button'))
    expect(handler).toHaveBeenCalledTimes(1)
  })

  test('does not call onClick when disabled', async () => {
    const handler = jest.fn()
    render(<ActionBtn label="Launch" color="#65c16a" disabled onClick={handler} />)
    await userEvent.click(screen.getByRole('button'))
    expect(handler).not.toHaveBeenCalled()
  })

  test('does not call onClick when busy', async () => {
    const handler = jest.fn()
    render(<ActionBtn label="Launch" color="#65c16a" busy onClick={handler} />)
    await userEvent.click(screen.getByRole('button'))
    expect(handler).not.toHaveBeenCalled()
  })

  test('button is disabled when disabled prop is true', () => {
    render(<ActionBtn label="Launch" color="#65c16a" disabled />)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  test('button is disabled when busy prop is true', () => {
    render(<ActionBtn label="Launch" color="#65c16a" busy />)
    expect(screen.getByRole('button')).toBeDisabled()
  })
})
