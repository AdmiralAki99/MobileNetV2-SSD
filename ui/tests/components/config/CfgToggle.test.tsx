import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CfgToggle } from '../../../src/components/config/CfgToggle'

describe('CfgToggle', () => {
  test('renders the label', () => {
    render(<CfgToggle label="Mixed precision" value={false} onChange={() => {}} />)
    expect(screen.getByText('Mixed precision')).toBeInTheDocument()
  })

  test('renders sub text when provided', () => {
    render(<CfgToggle label="Mixed precision" sub="fp16 + fp32" value={false} onChange={() => {}} />)
    expect(screen.getByText('fp16 + fp32')).toBeInTheDocument()
  })

  test('aria-checked is true when value is true', () => {
    render(<CfgToggle label="AMP" value={true} onChange={() => {}} />)
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
  })

  test('aria-checked is false when value is false', () => {
    render(<CfgToggle label="AMP" value={false} onChange={() => {}} />)
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'false')
  })

  test('calls onChange with toggled value on click', async () => {
    const handler = jest.fn()
    render(<CfgToggle label="AMP" value={false} onChange={handler} />)
    await userEvent.click(screen.getByRole('switch'))
    expect(handler).toHaveBeenCalledWith(true)
  })

  test('calls onChange with false when currently true', async () => {
    const handler = jest.fn()
    render(<CfgToggle label="AMP" value={true} onChange={handler} />)
    await userEvent.click(screen.getByRole('switch'))
    expect(handler).toHaveBeenCalledWith(false)
  })
})
