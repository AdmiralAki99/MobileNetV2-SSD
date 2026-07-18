# MobileNetV2 Backbone

`src/mobilenetv2ssd/models/mobilenet_v2/{blocks.py,backbone.py}`

## Why a custom backbone instead of `tf.keras.applications.MobileNetV2`

The SSD head needs intermediate feature maps (`C2`–`C5`) at specific strides, with
predictable layer names for weight transplant and export. `MobileNetV2` (subclass of
`tf.keras.Model`) hand-builds the same architecture as Keras's built-in one but exposes
a `feature_maps: dict[str, Tensor]` from `call()`, and starts from **random init +
transplanted ImageNet weights** rather than loading Keras's own weights file directly
(so residual/skip wiring matches this repo's exact block layout).

## Blocks (`blocks.py`)

- **`StandardConvolutionBlock`** — the stem: `Conv2D(3x3, stride) → BatchNorm → ReLU6`.
  Channel count scales by `alpha` (width multiplier).
- **`InvertedResidualBlock`** — the MobileNetV2 bottleneck: optional 1x1 expansion conv
  (skipped when `expansion_factor == 1`, i.e. the very first bottleneck) → depthwise 3x3
  conv → 1x1 projection conv, each with its own BN. Channel counts are computed lazily in
  `build()` (needs the input shape to know the expansion channel count), and rounded to a
  multiple of 8 via `_make_divisible` — matches the reference MobileNetV2's channel
  rounding so transplanted weights line up shape-for-shape.

Both blocks carry a `transplant_weights(reference_table, name_dict)` method used only
during first-time backbone construction (see below) — not used again once the local
`.weights.h5` cache exists.

## `MobileNetV2` model (`backbone.py`)

18 blocks total (`conv_1` stem + 17 inverted-residual bottlenecks + `conv_2` 1x1 head),
matching the reference architecture's `block_1`..`block_17` + `Conv_1`. Residual
(`Add()`) connections are applied manually in `call()` wherever stride=1 and channel
count is unchanged between consecutive same-config blocks (this mirrors the reference
model instead of encoding it as a per-block flag).

`call()` returns a `dict` of feature maps at four strides, keyed by their canonical
names — this is the contract the SSD head (`feature_maps` config key) reads from:

| Key | After block | Stride (300×300 input) |
|---|---|---|
| `C2` | bottleneck_3 (+ residual from bottleneck_2) | 8 |
| `C3` | bottleneck_6 | 16 |
| `C4` | bottleneck_13 | 32 |
| `C5` | conv_2 (final 1x1 + BN + ReLU6) | 32 (channel-widened, same spatial as C4) |

## Weight loading: transplant vs. cache (`build_backbone`)

```
build_backbone(input_shape, alpha, weights_dir)
  ├─ build_custom_mobilenetv2_backbone(...)   # random-init model, built via model(input_layer)
  ├─ if weights_dir/mobilenetv2_imagenet_notop_{W}x{H}_{alpha}.weights.h5 exists:
  │     └─ load_mobilenetv2_weights(model, path)   # model.load_weights(path)
  │        return model
  └─ else (first run for this input size / alpha):
        ├─ ref_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, ...)
        ├─ translation_map = create_reference_table()   # this-model-name -> Keras-reference-name
        ├─ model.transplant_weights(ref_layers, translation_map)   # copies conv/BN weights layer-by-layer
        ├─ model(tf.zeros((1, *input_shape)))            # force build so save_weights works (Keras 3)
        └─ model.save_weights(weights_path)               # cached for next time
```

`create_reference_table()` builds the name mapping programmatically
(`make_block_map(k, ref_idx)` for blocks 2–17, since they follow a uniform naming
pattern; block 1 and the stem/head are hardcoded because they're structurally
different — no expansion conv, or different names entirely).

**Gotcha this bit us once:** `build_custom_mobilenetv2_backbone` must call
`mobilenetv2(input_layer)` (Keras's `__call__`, which runs the build machinery) rather
than `mobilenetv2.call(input_layer)` (bypasses it) — otherwise `model.built` stays
`False` and the load-from-cache path fails with *"you are loading weights into a model
that has not yet been built"* under Keras 3, even though the transplant-and-save path
still works (it force-builds explicitly before saving). Fixed 2026-07-17.

The cache file is keyed by `(input_size, alpha)` in its filename — a new resolution or
width multiplier triggers one-time transplant + a new cache entry;
`src/mobilenetv2ssd/models/mobilenet_v2/weights/` is gitignored.

## `alpha` (width multiplier)

Standard MobileNetV2 scaling knob — all channel counts multiply by `alpha` (min-clamped,
then rounded to a multiple of 8). `alpha=1.0` is the default/only value currently
exercised in configs; lowering it would shrink the model for tighter embedded budgets at
the cost of accuracy, but changes the weights cache filename and forces a fresh
transplant (Keras's ImageNet weights are themselves only published for a few discrete
alpha values — `tf.keras.applications.MobileNetV2` will raise if you pick one it doesn't
have a weights file for).
