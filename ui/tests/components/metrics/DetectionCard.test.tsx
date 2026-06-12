import { render, screen, fireEvent } from '@testing-library/react'
import { DetectionCard, type DetectionImage } from '../../../src/components/metrics/DetectionCard'

const img: DetectionImage = {
  id: 0,
  label: 'aeroplane',
  boxes: [
    { x: 0.1, y: 0.1, w: 0.3, h: 0.4, cls: 'aeroplane', score: 0.92 },
    { x: 0.5, y: 0.2, w: 0.2, h: 0.3, cls: 'bird',      score: 0.74 },
  ],
}

describe('DetectionCard', () => {
  test('renders the card', () => {
    render(<DetectionCard img={img} />)
    expect(screen.getByTestId('detection-card-0')).toBeInTheDocument()
  })

  test('shows the image label', () => {
    render(<DetectionCard img={img} />)
    expect(screen.getByTestId('card-label-0')).toHaveTextContent('aeroplane')
  })

  test('shows detection count', () => {
    render(<DetectionCard img={img} />)
    expect(screen.getByTestId('card-count-0')).toHaveTextContent('2 det')
  })

  test('renders a group for each box', () => {
    const { container } = render(<DetectionCard img={img} />)
    expect(container.querySelectorAll('[data-testid^="box-0-"]').length).toBe(2)
  })

  test('hovering a box updates stroke width', () => {
    render(<DetectionCard img={img} />)
    const box = screen.getByTestId('box-0-0')
    const rect = box.querySelector('rect')!
    expect(rect.getAttribute('stroke-width')).toBe('1.3')
    fireEvent.mouseEnter(box)
    expect(rect.getAttribute('stroke-width')).toBe('2')
  })

  test('mouse leave restores stroke width', () => {
    render(<DetectionCard img={img} />)
    const box = screen.getByTestId('box-0-0')
    const rect = box.querySelector('rect')!
    fireEvent.mouseEnter(box)
    fireEvent.mouseLeave(box)
    expect(rect.getAttribute('stroke-width')).toBe('1.3')
  })
})
