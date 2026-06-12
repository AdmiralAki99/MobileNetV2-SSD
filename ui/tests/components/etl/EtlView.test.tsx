import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EtlView } from '../../../src/components/etl/EtlView'
import * as client from '../../../src/api/client'
import type { EtlVideo, EtlFrame, EtlAnnotation, EtlStats } from '../../../src/components/etl/etlTypes'

jest.mock('../../../src/api/client')
const mockClient = client as jest.Mocked<typeof client>

const videos: EtlVideo[] = [
  { id: 'v1', filename: 'clip_001.mp4', duration: 12.5, fps: 30, width: 1920, height: 1080, frames: 375, annotations: 840, status: 'completed' },
  { id: 'v2', filename: 'clip_002.mp4', duration: 8.0,  fps: 25, width: 1280, height: 720,  frames: 200, annotations: 310, status: 'processing' },
]

const frames: EtlFrame[] = [
  { id: 'f1', frame_index: 0,  timestamp_s: 0.0,  scene_change_score: 0.91, annotation_count: 3 },
  { id: 'f2', frame_index: 15, timestamp_s: 0.5,  scene_change_score: 0.12, annotation_count: 1 },
]

const annotations: EtlAnnotation[] = [
  { id: 'a1', class_name: 'person', x1: 0.1, y1: 0.2, x2: 0.3, y2: 0.8, votes: 3, consensus_score: 0.95,
    model_confidences: { yolov8: 0.92, rtdetr: 0.88, grounding_dino: 0.97 } },
]

const stats: EtlStats = {
  total_videos: 12, total_frames: 3400, total_annotations: 18200,
  class_distribution: [
    { class_name: 'person', count: 8000 },
    { class_name: 'vehicle', count: 4000 },
  ],
}

beforeEach(() => {
  mockClient.fetchEtlFrames.mockResolvedValue(frames)
  mockClient.fetchEtlAnnotations.mockResolvedValue(annotations)
})
afterEach(() => jest.clearAllMocks())

describe('EtlView', () => {
  test('renders the view', () => {
    render(<EtlView />)
    expect(screen.getByTestId('etl-view')).toBeInTheDocument()
  })

  test('shows dashes when no stats provided', () => {
    render(<EtlView />)
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(3)
  })

  test('renders summary stats when provided', () => {
    render(<EtlView statsData={stats} />)
    expect(screen.getByTestId('stat-videos')).toHaveTextContent('12')
    expect(screen.getByTestId('stat-frames')).toHaveTextContent('3,400')
    expect(screen.getByTestId('stat-annotations')).toHaveTextContent('18,200')
  })

  test('shows empty state when no videos', () => {
    render(<EtlView />)
    expect(screen.getByText('No videos processed yet.')).toBeInTheDocument()
  })

  test('renders a row for each video', () => {
    render(<EtlView videosData={videos} />)
    expect(screen.getByTestId('video-row-v1')).toBeInTheDocument()
    expect(screen.getByTestId('video-row-v2')).toBeInTheDocument()
  })

  test('renders video filenames', () => {
    render(<EtlView videosData={videos} />)
    expect(screen.getByText('clip_001.mp4')).toBeInTheDocument()
    expect(screen.getByText('clip_002.mp4')).toBeInTheDocument()
  })

  test('renders class distribution bars', () => {
    render(<EtlView statsData={stats} />)
    expect(screen.getByTestId('dist-row-person')).toBeInTheDocument()
    expect(screen.getByTestId('dist-row-vehicle')).toBeInTheDocument()
  })

  test('widest bar is 100% for max class', () => {
    render(<EtlView statsData={stats} />)
    expect(screen.getByTestId('dist-bar-person').style.width).toBe('100%')
  })

  test('second bar is proportional', () => {
    render(<EtlView statsData={stats} />)
    expect(screen.getByTestId('dist-bar-vehicle').style.width).toBe('50%')
  })

  test('shows empty class dist message when no dist data', () => {
    render(<EtlView statsData={{ ...stats, class_distribution: [] }} />)
    expect(screen.getByText('No annotations yet.')).toBeInTheDocument()
  })

  test('inspector not shown before selecting a video', () => {
    render(<EtlView videosData={videos} />)
    expect(screen.queryByTestId('frame-inspector')).not.toBeInTheDocument()
  })

  test('clicking a video opens the inspector', async () => {
    render(<EtlView videosData={videos} />)
    await userEvent.click(screen.getByTestId('video-row-v1'))
    expect(screen.getByTestId('frame-inspector')).toBeInTheDocument()
  })

  test('inspector shows the selected video filename', async () => {
    render(<EtlView videosData={videos} />)
    await userEvent.click(screen.getByTestId('video-row-v1'))
    expect(screen.getByTestId('frame-inspector')).toHaveTextContent('clip_001.mp4')
  })

  test('clicking close button hides the inspector', async () => {
    render(<EtlView videosData={videos} />)
    await userEvent.click(screen.getByTestId('video-row-v1'))
    await userEvent.click(screen.getByTestId('close-inspector'))
    expect(screen.queryByTestId('frame-inspector')).not.toBeInTheDocument()
  })

  test('frame list loads after selecting a video', async () => {
    render(<EtlView videosData={videos} />)
    await userEvent.click(screen.getByTestId('video-row-v1'))
    await waitFor(() => expect(screen.getByTestId('frame-row-f1')).toBeInTheDocument())
    expect(screen.getByTestId('frame-row-f2')).toBeInTheDocument()
  })

  test('annotation detail prompt shown before frame selection', async () => {
    render(<EtlView videosData={videos} />)
    await userEvent.click(screen.getByTestId('video-row-v1'))
    expect(screen.getByText('Select a frame to inspect annotations')).toBeInTheDocument()
  })

  test('selecting a frame loads annotations', async () => {
    render(<EtlView videosData={videos} />)
    await userEvent.click(screen.getByTestId('video-row-v1'))
    await waitFor(() => screen.getByTestId('frame-row-f1'))
    await userEvent.click(screen.getByTestId('frame-row-f1'))
    await waitFor(() => expect(screen.getByTestId('ann-row-a1')).toBeInTheDocument())
  })

  test('annotation row shows class name and vote count', async () => {
    render(<EtlView videosData={videos} />)
    await userEvent.click(screen.getByTestId('video-row-v1'))
    await waitFor(() => screen.getByTestId('frame-row-f1'))
    await userEvent.click(screen.getByTestId('frame-row-f1'))
    await waitFor(() => screen.getByTestId('ann-row-a1'))
    expect(screen.getByTestId('ann-row-a1')).toHaveTextContent('person')
    expect(screen.getByTestId('ann-row-a1')).toHaveTextContent('3/3')
  })
})
