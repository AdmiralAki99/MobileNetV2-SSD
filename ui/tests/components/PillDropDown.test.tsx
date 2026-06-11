import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PillDropdown } from '../../src/components/PillDropdown'

const options = ['us-east-1', 'us-west-2', 'ap-southeast-1']

describe('PillDropdown', () => {
  test('renders the current value', () => {
    render(<PillDropdown value="us-east-1" options={options} />)
    expect(screen.getByText('us-east-1')).toBeInTheDocument()
  })

  test('renders label when provided', () => {
    render(<PillDropdown label="Region" value="us-east-1" options={options} />)
    expect(screen.getByText('Region')).toBeInTheDocument()
  })

  test('dropdown is closed by default', () => {
    render(<PillDropdown value="us-east-1" options={options} />)
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
  })

  test('opens dropdown on button click', async () => {
    render(<PillDropdown value="us-east-1" options={options} />)
    await userEvent.click(screen.getByRole('button'))
    expect(screen.getByRole('listbox')).toBeInTheDocument()
  })

  test('renders all options when open', async () => {
    render(<PillDropdown value="us-east-1" options={options} />)
    await userEvent.click(screen.getByRole('button'))
    options.forEach(o => expect(screen.getAllByText(o).length).toBeGreaterThanOrEqual(1))
  })

  test('selecting an option updates the value and closes dropdown', async () => {
    render(<PillDropdown value="us-east-1" options={options} />)
    await userEvent.click(screen.getByRole('button'))
    await userEvent.click(screen.getByText('us-west-2'))
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
    expect(screen.getByText('us-west-2')).toBeInTheDocument()
  })

  test('closes dropdown on outside click', async () => {
    render(
      <div>
        <PillDropdown value="us-east-1" options={options} />
        <button>outside</button>
      </div>
    )
    await userEvent.click(screen.getByRole('button', { name: /us-east-1/i }))
    expect(screen.getByRole('listbox')).toBeInTheDocument()
    await userEvent.click(screen.getByText('outside'))
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
  })
})
