export interface Config {
  experiment_id: string
  backbone: string
  pretrained: boolean
  input_size: number
  num_classes: number
  optimizer: string
  learning_rate: number
  lr_schedule: string
  warmup_epochs: number
  batch_size: number
  epochs: number
  use_amp: boolean
  grad_clip: number
  cls_loss: string
  loc_loss: string
  focal_alpha: number
  focal_gamma: number
  neg_pos_ratio: number
  dataset: string
  aug_flip: boolean
  aug_color: boolean
  aug_crop: boolean
  aug_scale: boolean
  instance_type: string
  region: string
  spot: boolean
}

export const DEFAULT_CFG: Config = {
  experiment_id: 'exp005_custom',
  backbone: 'mobilenetv2', pretrained: true, input_size: 300, num_classes: 21,
  optimizer: 'adam', learning_rate: 1e-3, lr_schedule: 'cosine',
  warmup_epochs: 5, batch_size: 32, epochs: 200, use_amp: true, grad_clip: 1.0,
  cls_loss: 'focal', loc_loss: 'smooth_l1', focal_alpha: 0.25, focal_gamma: 2.0, neg_pos_ratio: 3,
  dataset: 'voc2012', aug_flip: true, aug_color: true, aug_crop: true, aug_scale: false,
  instance_type: 'p3.2xlarge', region: 'us-east-1', spot: false,
}

export const toYAML = (cfg: Config): string => [
  `# sentinel> config builder`,
  `experiment_id: ${cfg.experiment_id}`,
  ``,
  `model:`,
  `  backbone: ${cfg.backbone}`,
  `  pretrained: ${cfg.pretrained}`,
  `  input_size: [${cfg.input_size}, ${cfg.input_size}, 3]`,
  `  num_classes: ${cfg.num_classes}`,
  ``,
  `training:`,
  `  optimizer: ${cfg.optimizer}`,
  `  learning_rate: ${cfg.learning_rate.toExponential(1)}`,
  `  lr_schedule: ${cfg.lr_schedule}`,
  `  warmup_epochs: ${cfg.warmup_epochs}`,
  `  batch_size: ${cfg.batch_size}`,
  `  epochs: ${cfg.epochs}`,
  `  use_amp: ${cfg.use_amp}`,
  `  grad_clip: ${cfg.grad_clip}`,
  ``,
  `loss:`,
  `  classification: ${cfg.cls_loss}`,
  `  localization: ${cfg.loc_loss}`,
  ...(cfg.cls_loss === 'focal' ? [`  focal:`, `    alpha: ${cfg.focal_alpha}`, `    gamma: ${cfg.focal_gamma}`] : []),
  `  neg_pos_ratio: ${cfg.neg_pos_ratio}`,
  ``,
  `data:`,
  `  dataset: ${cfg.dataset}`,
  `  augmentations:`,
  `    horizontal_flip: ${cfg.aug_flip}`,
  `    color_jitter: ${cfg.aug_color}`,
  `    random_crop: ${cfg.aug_crop}`,
  `    random_scale: ${cfg.aug_scale}`,
  ``,
  `deploy:`,
  `  instance_type: ${cfg.instance_type}`,
  `  region: ${cfg.region}`,
  `  spot_instance: ${cfg.spot}`,
].join('\n')
