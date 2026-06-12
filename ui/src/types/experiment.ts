export interface Experiment {
  experiment_id: string
  fingerprint?: string
  status: string
  region?: string
  ec2_instance?: string
  best_metric?: number
  best_epoch?: number
  total_steps?: number
  checkpoint_s3_path?: string
  config_filename?: string
  completed_at?: string
  claimed?: string
  failure_reason?: string
}
