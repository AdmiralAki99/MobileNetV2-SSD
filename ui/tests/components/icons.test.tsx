import { render } from '@testing-library/react'
import {
  ChevronDown, ChevronUp, ArrowLeft, ArrowUp,
  SearchSvg, GearSvg, RefreshSvg, MoreSvg,
  PlusSvg, CloseSvg, FilterSvg, CopySvg,
  UserSvg, LayersSvg, ShieldCheck, DangerTriangle,
  AlertCircle, UnknownSquare,
} from '../../src/components/icons'

const icons = [
  ChevronDown, ChevronUp, ArrowLeft, ArrowUp,
  SearchSvg, GearSvg, RefreshSvg, MoreSvg,
  PlusSvg, CloseSvg, FilterSvg, CopySvg,
  UserSvg, LayersSvg, ShieldCheck, DangerTriangle,
  AlertCircle, UnknownSquare,
]

describe('Icons', () => {
  test.each(icons)('%s renders an svg without crashing', (IconComponent) => {
    const { container } = render(<IconComponent />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  test('Icon respects size prop', () => {
    const { container } = render(<ChevronDown size={24} />)
    const svg = container.querySelector('svg')!
    expect(svg.getAttribute('width')).toBe('24')
    expect(svg.getAttribute('height')).toBe('24')
  })

  test('ShieldCheck respects color prop', () => {
    const { container } = render(<ShieldCheck color="#ff0000" />)
    const path = container.querySelector('path')!
    expect(path.getAttribute('fill')).toBe('#ff0000')
  })

  test('DangerTriangle respects color prop', () => {
    const { container } = render(<DangerTriangle color="#ff0000" />)
    const path = container.querySelector('path')!
    expect(path.getAttribute('fill')).toBe('#ff0000')
  })
})
