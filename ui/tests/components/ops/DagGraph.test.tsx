import { render, screen } from '@testing-library/react'
import { DagGraph, type Task } from '../../../src/components/ops/DagGraph'

const tasks: Task[] = [
  { task_id: 'provision_ec2', state: 'success', duration: 0.8,   start_date: '2026-05-23T02:00:02Z' },
  { task_id: 'wait_for_ray',  state: 'success', duration: 12.3,  start_date: '2026-05-23T02:00:03Z' },
  { task_id: 'run_etl_job',   state: 'running', duration: null,  start_date: '2026-05-23T02:00:16Z' },
]

describe('DagGraph', () => {
  test('renders the graph', () => {
    render(<DagGraph tasks={tasks} />)
    expect(screen.getByTestId('dag-graph')).toBeInTheDocument()
  })

  test('renders a node for each task', () => {
    render(<DagGraph tasks={tasks} />)
    tasks.forEach(t => expect(screen.getByTestId(`task-node-${t.task_id}`)).toBeInTheDocument())
  })

  test('renders task ids as text', () => {
    render(<DagGraph tasks={tasks} />)
    expect(screen.getByText('provision_ec2')).toBeInTheDocument()
    expect(screen.getByText('wait_for_ray')).toBeInTheDocument()
  })

  test('shows empty message when tasks is empty', () => {
    render(<DagGraph tasks={[]} />)
    expect(screen.getByText('No tasks for this run.')).toBeInTheDocument()
  })

  test('renders — for null duration', () => {
    render(<DagGraph tasks={tasks} />)
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })
})
