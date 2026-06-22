export interface ConfigItem {
  name: string
  path: string
  content: Record<string, unknown>
}

export interface ConfigLibrary {
  backbones?: ConfigItem[]
  losses?: ConfigItem[]
  optimizers?: ConfigItem[]
  augmentations?: ConfigItem[]
  heads?: ConfigItem[]
  priors?: ConfigItem[]
  eval?: ConfigItem[]
  train?: ConfigItem[]
  samplers?: ConfigItem[]
  checkpoint?: ConfigItem[]
  logging?: ConfigItem[]
  runtime?: ConfigItem[]
  export?: ConfigItem[]
  [key: string]: ConfigItem[] | undefined
}

export interface RegisterRequest {
  config: ExperimentConfig
  task_type: string
  git_commit: null
}

export interface RegisterResponse {
  experiment_id: string
  fingerprint: string
  config_ref: string
  created: boolean
}

export interface ExperimentConfig {
  experiment: {
    id: string
    name: string
    description: string
    author: string
    tags: string[]
    priority: number
    depends_on: string[]
  }
  infrastructure: {
    instance_type: string
    spot: boolean
    region: string
    dynamodb_table: string
  }
  defaults: Record<string, string>
  input_size: [number, number]
  num_classes: number
  overrides: Record<string, unknown>
}

export const CATEGORY_ORDER = [
  'backbones', 'losses', 'optimizers', 'augmentations',
  'heads', 'priors', 'train', 'eval',
  'samplers', 'checkpoint', 'logging', 'runtime', 'export',
]

export const CATEGORY_LABEL: Record<string, string> = {
  backbones: 'Backbone', losses: 'Loss', optimizers: 'Optimizer',
  augmentations: 'Augmentation', heads: 'Heads', priors: 'Priors',
  train: 'Train', eval: 'Eval', samplers: 'Sampler',
  checkpoint: 'Checkpoint', logging: 'Logging', runtime: 'Runtime', export: 'Export',
}
