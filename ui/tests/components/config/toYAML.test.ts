import { toYAML, DEFAULT_CFG } from '../../../src/components/config/configTypes'

describe('toYAML', () => {
  test('contains experiment_id', () => {
    const yaml = toYAML(DEFAULT_CFG)
    expect(yaml).toContain('experiment_id: exp005_custom')
  })

  test('contains backbone', () => {
    const yaml = toYAML(DEFAULT_CFG)
    expect(yaml).toContain('backbone: mobilenetv2')
  })

  test('contains optimizer', () => {
    const yaml = toYAML(DEFAULT_CFG)
    expect(yaml).toContain('optimizer: adam')
  })

  test('contains focal loss params when cls_loss is focal', () => {
    const yaml = toYAML({ ...DEFAULT_CFG, cls_loss: 'focal' })
    expect(yaml).toContain('focal:')
    expect(yaml).toContain('alpha: 0.25')
    expect(yaml).toContain('gamma: 2')
  })

  test('omits focal params when cls_loss is not focal', () => {
    const yaml = toYAML({ ...DEFAULT_CFG, cls_loss: 'cross_entropy' })
    expect(yaml).not.toContain('focal:')
  })

  test('reflects custom experiment_id', () => {
    const yaml = toYAML({ ...DEFAULT_CFG, experiment_id: 'my_custom_exp' })
    expect(yaml).toContain('experiment_id: my_custom_exp')
  })

  test('reflects spot instance setting', () => {
    expect(toYAML({ ...DEFAULT_CFG, spot: true })).toContain('spot_instance: true')
    expect(toYAML({ ...DEFAULT_CFG, spot: false })).toContain('spot_instance: false')
  })

  test('includes augmentation flags', () => {
    const yaml = toYAML({ ...DEFAULT_CFG, aug_flip: true, aug_scale: false })
    expect(yaml).toContain('horizontal_flip: true')
    expect(yaml).toContain('random_scale: false')
  })
})
