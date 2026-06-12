import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfigView } from '../../../src/components/config/ConfigView'
import * as client from '../../../src/api/client'

jest.mock('../../../src/api/client')
const mockClient = client as jest.Mocked<typeof client>

beforeEach(() => {
  mockClient.launchTraining.mockResolvedValue({ status: 'ok', message: 'Launched' })
})
afterEach(() => jest.clearAllMocks())

describe('ConfigView', () => {
  test('renders the view', () => {
    render(<ConfigView />)
    expect(screen.getByTestId('config-view')).toBeInTheDocument()
  })

  test('renders yaml preview', () => {
    render(<ConfigView />)
    expect(screen.getByTestId('yaml-preview')).toBeInTheDocument()
  })

  test('yaml contains default experiment id', () => {
    render(<ConfigView />)
    expect(screen.getByTestId('yaml-preview')).toHaveTextContent('exp005_custom')
  })

  test('changing experiment id updates yaml', async () => {
    render(<ConfigView />)
    const input = screen.getByPlaceholderText('exp005_my_run')
    await userEvent.clear(input)
    await userEvent.type(input, 'my_new_exp')
    expect(screen.getByTestId('yaml-preview')).toHaveTextContent('my_new_exp')
  })

  test('launch button opens modal', async () => {
    render(<ConfigView />)
    await userEvent.click(screen.getByText('Launch Experiment'))
    expect(screen.getByTestId('launch-modal')).toBeInTheDocument()
  })

  test('cancel button closes modal', async () => {
    render(<ConfigView />)
    await userEvent.click(screen.getByText('Launch Experiment'))
    await userEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByTestId('launch-modal')).not.toBeInTheDocument()
  })

  test('confirm launch calls launchTraining', async () => {
    render(<ConfigView />)
    await userEvent.click(screen.getByText('Launch Experiment'))
    await userEvent.click(screen.getByTestId('confirm-launch'))
    await waitFor(() => expect(mockClient.launchTraining).toHaveBeenCalled())
  })

  test('shows success banner after launch', async () => {
    render(<ConfigView />)
    await userEvent.click(screen.getByText('Launch Experiment'))
    await userEvent.click(screen.getByTestId('confirm-launch'))
    await waitFor(() => expect(screen.getByTestId('launch-banner')).toBeInTheDocument())
  })

  test('success banner shows fingerprint', async () => {
    render(<ConfigView />)
    await userEvent.click(screen.getByText('Launch Experiment'))
    await userEvent.click(screen.getByTestId('confirm-launch'))
    await waitFor(() => expect(screen.getByTestId('launch-fingerprint')).toHaveTextContent('fp ·'))
  })

  test('dismissing banner hides it', async () => {
    render(<ConfigView />)
    await userEvent.click(screen.getByText('Launch Experiment'))
    await userEvent.click(screen.getByTestId('confirm-launch'))
    await waitFor(() => screen.getByTestId('launch-banner'))
    await userEvent.click(screen.getByText('✕'))
    expect(screen.queryByTestId('launch-banner')).not.toBeInTheDocument()
  })

  test('toggling augmentation updates yaml', async () => {
    render(<ConfigView />)
    const toggle = screen.getByRole('switch', { name: 'Random scale' })
    expect(screen.getByTestId('yaml-preview')).toHaveTextContent('random_scale: false')
    await userEvent.click(toggle)
    expect(screen.getByTestId('yaml-preview')).toHaveTextContent('random_scale: true')
  })
})
