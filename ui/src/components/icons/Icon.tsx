interface IconProps {
  children: React.ReactNode
  size?: number
  [key: string]: unknown
}

export const Icon = ({ children, size = 16, ...props }: IconProps) => (
  <svg
    width={size} height={size} viewBox="0 0 16 16"
    fill="none" stroke="currentColor"
    strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"
    {...props}
  >
    {children}
  </svg>
)
