import { Icon } from './Icon'

export const ChevronDown  = ({ size = 12 }) => <Icon size={size}><path d="M4 6l4 4 4-4"/></Icon>
export const ChevronUp    = ({ size = 12 }) => <Icon size={size}><path d="M4 10l4-4 4 4"/></Icon>
export const ArrowLeft    = ({ size = 16 }) => <Icon size={size}><path d="M10 3L5 8l5 5"/></Icon>
export const ArrowUp      = ({ size = 14 }) => <Icon size={size}><path d="M8 12V4M4 7l4-3 4 3"/></Icon>
export const SearchSvg    = ({ size = 15 }) => <Icon size={size}><circle cx="6.8" cy="6.8" r="4.2"/><path d="M10 10l3.5 3.5"/></Icon>
export const GearSvg      = ({ size = 17 }) => <Icon size={size} strokeWidth="1.2"><circle cx="8" cy="8" r="2.2"/><path d="M8 2v1.5M8 12.5V14M2 8h1.5M12.5 8H14M3.8 3.8l1 1M11.2 11.2l1 1M3.8 12.2l1-1M11.2 4.8l1-1"/></Icon>
export const RefreshSvg   = ({ size = 14 }) => <Icon size={size}><path d="M2.5 8a5.5 5.5 0 019.2-3M13.5 8a5.5 5.5 0 01-9.2 3"/><path d="M11.7 2v3h-3M4.3 14v-3h3" strokeWidth="1.2"/></Icon>
export const MoreSvg      = ({ size = 14 }) => <Icon size={size} fill="currentColor" stroke="none"><circle cx="3" cy="8" r="1.2"/><circle cx="8" cy="8" r="1.2"/><circle cx="13" cy="8" r="1.2"/></Icon>
export const PlusSvg      = ({ size = 14 }) => <Icon size={size}><path d="M8 3v10M3 8h10"/></Icon>
export const CloseSvg     = ({ size = 12 }) => <Icon size={size}><path d="M3 3l10 10M13 3L3 13"/></Icon>
export const FilterSvg    = ({ size = 14 }) => <Icon size={size}><path d="M2 4h12M4 8h8M6 12h4"/></Icon>
export const CopySvg      = ({ size = 14 }) => <Icon size={size}><rect x="5" y="5" width="8" height="8" rx="1.5"/><path d="M3 11V3h8"/></Icon>
export const UserSvg      = ({ size = 14 }) => <Icon size={size}><circle cx="8" cy="5" r="2.5"/><path d="M3 14c0-2.8 2.2-5 5-5s5 2.2 5 5"/></Icon>
export const LayersSvg    = ({ size = 14 }) => <Icon size={size}><path d="M8 2L2 5.5 8 9l6-3.5L8 2z"/><path d="M2 8.5L8 12l6-3.5" strokeWidth="1.2"/><path d="M2 11.5L8 15l6-3.5" strokeWidth="1.2"/></Icon>

export const ShieldCheck = ({ size = 16, color = '#65c16a' }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <path d="M8 1.5L2.5 4v3.5c0 3.8 2.3 6.2 5.5 7 3.2-.8 5.5-3.2 5.5-7V4L8 1.5z" fill={color} opacity="0.9"/>
    <path d="M6 8.5l1.5 1.5L10 7" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
)

export const DangerTriangle = ({ size = 16, color = '#e84855' }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <path d="M8 2L1.5 13.5h13L8 2z" fill={color} opacity="0.9"/>
    <path d="M8 6.5v3" stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
    <circle cx="8" cy="11.2" r="0.7" fill="#fff"/>
  </svg>
)

export const AlertCircle = ({ size = 16, color = '#e88548' }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <circle cx="8" cy="8" r="6.5" fill={color} opacity="0.9"/>
    <path d="M8 5v3.5" stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
    <circle cx="8" cy="10.8" r="0.7" fill="#fff"/>
  </svg>
)

export const UnknownSquare = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <rect x="2.5" y="2.5" width="11" height="11" rx="2" fill="#555" opacity="0.7"/>
    <path d="M6.5 6a1.5 1.5 0 013 0c0 1-1.5 1-1.5 2" stroke="#fff" strokeWidth="1.2" strokeLinecap="round"/>
    <circle cx="8" cy="10.5" r="0.6" fill="#fff"/>
  </svg>
)
