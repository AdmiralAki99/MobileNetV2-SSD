import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfigView } from '../../../src/components/config/ConfigView'
import * as client from '../../../src/api/client'

jest.mock('../../../src/api/client')
const mockClient = client as jest.Mocked<typeof client>

const library = {
  backbones: [
    { name: 'mobilenetv2', path: 'base/backbones/mobilenetv2.yaml', content: { backbone: { name: 'mobilenetv2', width_mult: 1.0 } } },
  ],
  data: [
    { name: 'voc', path: 'data/voc.yaml', content: { dataset_name: 'voc', num_classes: 21 } },
  ],
  augmentations: [
    { name: 'ssd_augment', path: 'base/augmentations/ssd_augment.yaml', content: { horizontal_flip: true } },
  ],
  optimizers: [
    { name: 'sgd_cosine', path: 'base/optimizers/sgd_cosine.yaml', content: { optimizer: { name: 'sgd', lr: 0.001 } } },
  ],
  losses: [
    { name: 'ssd_loss', path: 'base/losses/ssd_loss.yaml', content: { classification: { type: 'cross_entropy' } } },
  ],
  priors: [
    { name: 'ssd_300', path: 'base/priors/ssd_300.yaml', content: { image_size: [300, 300], min_scale: 0.2, max_scale: 0.95 } },
  ],
}

function makeDataTransfer() {
  const store: Record<string, string> = {}
  return {
    setData: (k: string, v: string) => { store[k] = v },
    getData: (k: string) => store[k] ?? '',
    effectAllowed: '',
    dropEffect: '',
  } as unknown as DataTransfer
}

function dragCardToSlot(cardTestId: string, slotTestId: string) {
  const dt = makeDataTransfer()
  fireEvent.dragStart(screen.getByTestId(cardTestId), { dataTransfer: dt })
  fireEvent.dragOver(screen.getByTestId(slotTestId), { dataTransfer: dt })
  fireEvent.drop(screen.getByTestId(slotTestId), { dataTransfer: dt })
}

async function fillRequiredSlots() {
  dragCardToSlot('palette-item-backbone__mobilenetv2', 'slot-backbone')
  dragCardToSlot('palette-item-data__voc', 'slot-data')
  dragCardToSlot('palette-item-augmentation__ssd_augment', 'slot-augmentation')
  dragCardToSlot('palette-item-optimizer__sgd_cosine', 'slot-optimizer')
  dragCardToSlot('palette-item-loss__ssd_loss', 'slot-loss')
  dragCardToSlot('palette-item-priors__ssd_300', 'slot-priors')
  await waitFor(() => expect(screen.getByText('Launch')).not.toBeDisabled())
}

beforeEach(() => {
  mockClient.fetchConfigLibrary.mockResolvedValue(library as any)
  mockClient.refreshConfigLibrary.mockResolvedValue({ status: 'synced' })
  mockClient.registerExperiment.mockResolvedValue({
    experiment_id: 'exp005_custom', fingerprint: 'abc123ef', config_ref: 'experiments/exp005_custom.yaml', created: true,
  })
  mockClient.launchTraining.mockResolvedValue({ status: 200, dag_run_id: 'run_001' })
})
afterEach(() => jest.clearAllMocks())

describe('ConfigView', () => {
  test('renders the view', async () => {
    render(<ConfigView />)
    expect(screen.getByTestId('config-view')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
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
    const input = screen.getByPlaceholderText('exp005_custom')
    await userEvent.clear(input)
    await userEvent.type(input, 'my_new_exp')
    expect(screen.getByTestId('yaml-preview')).toHaveTextContent('my_new_exp')
  })

  test('launch is disabled and shows missing slots', async () => {
    render(<ConfigView />)
    expect(screen.getByText('Launch')).toBeDisabled()
    expect(screen.getByText(/needs:/)).toBeInTheDocument()
  })

  test('dragging a card into a slot fills it', async () => {
    render(<ConfigView />)
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
    dragCardToSlot('palette-item-backbone__mobilenetv2', 'slot-backbone')
    expect(screen.getByTestId('slot-backbone')).toHaveTextContent('Mobilenetv2')
  })

  test('filling all required slots enables launch', async () => {
    render(<ConfigView />)
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
    await fillRequiredSlots()
  })

  test('launch button opens modal once required slots are filled', async () => {
    render(<ConfigView />)
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
    await fillRequiredSlots()
    await userEvent.click(screen.getByText('Launch'))
    expect(screen.getByTestId('launch-modal')).toBeInTheDocument()
    expect(screen.getByText('Launch Experiment')).toBeInTheDocument()
  })

  test('cancel button closes modal', async () => {
    render(<ConfigView />)
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
    await fillRequiredSlots()
    await userEvent.click(screen.getByText('Launch'))
    await userEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByTestId('launch-modal')).not.toBeInTheDocument()
  })

  test('confirm launch registers then launches training', async () => {
    render(<ConfigView />)
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
    await fillRequiredSlots()
    await userEvent.click(screen.getByText('Launch'))
    await userEvent.click(screen.getByTestId('confirm-launch'))
    await waitFor(() => expect(mockClient.registerExperiment).toHaveBeenCalled())
    expect(mockClient.launchTraining).toHaveBeenCalledWith({ experiment_id: 'exp005_custom', fingerprint: 'abc123ef' })
  })

  test('shows success banner with fingerprint after launch', async () => {
    render(<ConfigView />)
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
    await fillRequiredSlots()
    await userEvent.click(screen.getByText('Launch'))
    await userEvent.click(screen.getByTestId('confirm-launch'))
    await waitFor(() => expect(screen.getByTestId('launch-banner')).toBeInTheDocument())
    expect(screen.getByTestId('launch-fingerprint')).toHaveTextContent('fp · abc123ef')
  })

  test('dismissing banner hides it', async () => {
    render(<ConfigView />)
    await waitFor(() => expect(screen.getByTestId('palette-item-backbone__mobilenetv2')).toBeInTheDocument())
    await fillRequiredSlots()
    await userEvent.click(screen.getByText('Launch'))
    await userEvent.click(screen.getByTestId('confirm-launch'))
    await waitFor(() => screen.getByTestId('launch-banner'))
    await userEvent.click(within(screen.getByTestId('launch-banner')).getByText('×'))
    expect(screen.queryByTestId('launch-banner')).not.toBeInTheDocument()
  })

  test('editing the yaml directly marks it dirty and offers a sync option', async () => {
    render(<ConfigView />)
    const editor = screen.getByTestId('yaml-editor')
    await userEvent.type(editor, '\nextra: true')
    expect(screen.getByText('sync from builder')).toBeInTheDocument()
  })

  test('new component modal opens and closes', async () => {
    render(<ConfigView />)
    await userEvent.click(screen.getByTitle('New component'))
    expect(screen.getByText('New Component')).toBeInTheDocument()
    await userEvent.click(screen.getByText('×', { selector: 'button' }))
    expect(screen.queryByText('New Component')).not.toBeInTheDocument()
  })
})
