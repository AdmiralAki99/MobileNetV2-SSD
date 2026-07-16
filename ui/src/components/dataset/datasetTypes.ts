export interface LedgerEntry {
  name: string
  split: string
  num_images: number
  num_boxes: number
}

export interface BoxDimsResponse {
  norm: [number, number][]
}

export interface LaunchResponse {
  status: number
  dag_run_id: string
}
