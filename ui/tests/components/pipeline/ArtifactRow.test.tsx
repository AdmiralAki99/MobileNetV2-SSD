import { render, screen } from '@testing-library/react'
import { ArtifactRow } from '../../../src/components/pipeline/ArtifactRow'

describe('ArtifactRow', () => {
  test('renders the label', () => {
    render(<ArtifactRow label="SavedModel" ok={true} />)
    expect(screen.getByText('SavedModel')).toBeInTheDocument()
  })

  test('renders a check icon when ok=true', () => {
    const { container } = render(<ArtifactRow label="SavedModel" ok={true} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  test('renders an alert icon when ok=false', () => {
    const { container } = render(<ArtifactRow label="ONNX" ok={false} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  test('label is primary color when ok=true', () => {
    const { container } = render(<ArtifactRow label="SavedModel" ok={true} />)
    const span = container.querySelector('span')!
    expect(span.style.color).toBe('var(--text-primary)')
  })

  test('label is tertiary color when ok=false', () => {
    const { container } = render(<ArtifactRow label="ONNX" ok={false} />)
    const span = container.querySelector('span')!
    expect(span.style.color).toBe('var(--text-tertiary)')
  })
})
