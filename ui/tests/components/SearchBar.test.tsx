import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SearchBar } from '../../src/components/SearchBar'

describe('SearchBar', () => {
  test('renders the input with placeholder', () => {
    render(<SearchBar />)
    expect(screen.getByPlaceholderText('Search by Name etc')).toBeInTheDocument()
  })

  test('renders the search icon svg', () => {
    const { container } = render(<SearchBar />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  test('border changes to accent color on focus', async () => {
    const { container } = render(<SearchBar />)
    const wrapper = container.firstChild as HTMLElement
    const input = screen.getByPlaceholderText('Search by Name etc')

    expect(wrapper.style.border).toContain('var(--border-subtle)')
    await userEvent.click(input)
    expect(wrapper.style.border).toContain('var(--accent)')
  })

  test('border reverts to subtle on blur', async () => {
    const { container } = render(<SearchBar />)
    const wrapper = container.firstChild as HTMLElement
    const input = screen.getByPlaceholderText('Search by Name etc')

    await userEvent.click(input)
    await userEvent.tab()
    expect(wrapper.style.border).toContain('var(--border-subtle)')
  })

  test('accepts typed input', async () => {
    render(<SearchBar />)
    const input = screen.getByPlaceholderText('Search by Name etc') as HTMLInputElement
    await userEvent.type(input, 'exp002')
    expect(input.value).toBe('exp002')
  })
})
