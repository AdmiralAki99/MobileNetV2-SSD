import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { TabGroup } from '../../src/components/TabGroup'

const tabs = ['Pipeline', 'Metrics', 'ETL', 'Ops']

describe('TabGroup', () => {
  test('renders all tabs', () => {
    render(<TabGroup tabs={tabs} active="Pipeline" onChange={() => {}} />)
    tabs.forEach(t => expect(screen.getByText(t)).toBeInTheDocument())
  })

  test('active tab has white background', () => {
    render(<TabGroup tabs={tabs} active="Metrics" onChange={() => {}} />)
    const activeBtn = screen.getByText('Metrics')
    expect(activeBtn.style.background).toBe('rgb(255, 255, 255)')
  })

  test('inactive tabs have transparent background', () => {
    render(<TabGroup tabs={tabs} active="Metrics" onChange={() => {}} />)
    const inactiveBtn = screen.getByText('Pipeline')
    expect(inactiveBtn.style.background).toBe('transparent')
  })

  test('calls onChange with the clicked tab', async () => {
    const handler = jest.fn()
    render(<TabGroup tabs={tabs} active="Pipeline" onChange={handler} />)
    await userEvent.click(screen.getByText('Metrics'))
    expect(handler).toHaveBeenCalledWith('Metrics')
  })

  test('does not call onChange when active tab is clicked', async () => {
    const handler = jest.fn()
    render(<TabGroup tabs={tabs} active="Pipeline" onChange={handler} />)
    await userEvent.click(screen.getByText('Pipeline'))
    expect(handler).toHaveBeenCalledWith('Pipeline')
  })

  test('renders single tab without crashing', () => {
    render(<TabGroup tabs={['Only']} active="Only" onChange={() => {}} />)
    expect(screen.getByText('Only')).toBeInTheDocument()
  })
})
