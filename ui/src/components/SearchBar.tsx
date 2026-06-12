import { useState } from 'react'
import { SearchSvg } from './icons'

export const SearchBar = () => {
  const [focused, setFocused] = useState(false)
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '7px 14px', borderRadius: 999, minWidth: 180,
      background: 'var(--bg-pill)',
      border: `1px solid ${focused ? 'var(--accent)' : 'var(--border-subtle)'}`,
      transition: 'border-color 0.25s',
    }}>
      <SearchSvg />
      <input
        placeholder="Search by Name etc"
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        style={{
          background: 'none', border: 'none', outline: 'none',
          color: 'var(--text-primary)', fontSize: '12.5px',
          fontFamily: 'inherit', width: '100%',
        }}
      />
    </div>
  )
}
