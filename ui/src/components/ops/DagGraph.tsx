import { TASK_COLOR, fmtDur } from './utils'

export interface Task {
  task_id: string
  state: string
  duration?: number | null
  start_date?: string
}

interface Props {
  tasks: Task[]
}

const BOX_W = 118, BOX_H = 54, ARROW_GAP = 28, PAD_X = 14, PAD_Y = 13

export const DagGraph = ({ tasks }: Props) => {
  if (tasks.length === 0) {
    return (
      <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '12px 4px' }}>
        No tasks for this run.
      </div>
    )
  }

  const svgW = tasks.length * BOX_W + (tasks.length - 1) * ARROW_GAP + PAD_X * 2
  const svgH = BOX_H + PAD_Y * 2

  return (
    <div data-testid="dag-graph" style={{ overflowX: 'auto' }}>
      <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} style={{ display: 'block' }}>
        {tasks.map((task, i) => {
          const x = PAD_X + i * (BOX_W + ARROW_GAP)
          const y = PAD_Y, cx = x + BOX_W / 2, cy = y + BOX_H / 2
          const color = TASK_COLOR[task.state] ?? '#494e4d'
          return (
            <g key={task.task_id} data-testid={`task-node-${task.task_id}`}>
              {i < tasks.length - 1 && (
                <line x1={x + BOX_W} y1={cy} x2={x + BOX_W + ARROW_GAP - 6} y2={cy}
                  stroke="rgba(255,255,255,0.12)" strokeWidth="1.5" />
              )}
              <rect x={x} y={y} width={BOX_W} height={BOX_H} rx="2"
                fill={`${color}10`} stroke={`${color}44`} strokeWidth="1" />
              <text x={cx} y={y + 17} textAnchor="middle" fill="rgba(232,234,233,0.85)"
                fontSize="9.5" fontFamily="monospace" fontWeight="600">{task.task_id}</text>
              <text x={cx} y={y + 31} textAnchor="middle" fill={color}
                fontSize="9" fontWeight="700">{(task.state ?? '').toUpperCase()}</text>
              <text x={cx} y={y + 45} textAnchor="middle" fill="#494e4d"
                fontSize="9" fontFamily="monospace">{fmtDur(task.duration)}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
