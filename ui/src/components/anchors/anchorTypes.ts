export interface DatasetEntry {
  name: string
  split: string
  num_images: number
  num_boxes: number
}

export interface ClusterResult {
  dataset: string
  min_scale: number
  max_scale: number
  aspect_ratios: number[]
  fitness: { mean_iou: number; 'recall@0.5': number }
  centroids: number[][]
}

export interface DeriveResponse {
  status: number
  result: ClusterResult
}

export interface BoxDims {
  norm: [number, number][]
}
