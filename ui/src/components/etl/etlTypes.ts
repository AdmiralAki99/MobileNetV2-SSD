export interface EtlVideo {
  id: string
  filename: string
  duration: number
  fps: number
  width: number
  height: number
  frames: number
  annotations: number
  status: string
}

export interface EtlFrame {
  id: string
  frame_index: number
  timestamp_s: number
  scene_change_score: number
  annotation_count: number
}

export interface EtlAnnotation {
  id: string
  class_name: string
  x1: number; y1: number; x2: number; y2: number
  votes: number
  consensus_score: number
  model_confidences: Record<string, number | null>
}

export interface EtlStats {
  total_videos?: number
  total_frames?: number
  total_annotations?: number
  class_distribution?: { class_name: string; count: number }[]
}
