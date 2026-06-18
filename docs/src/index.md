```@meta
CurrentModule = Tylo
```

# Tylo

Documentation for [Tylo](https://github.com/jool-space/Tylo.jl).

## Attention

```@docs
attention
attention!
∇attention
∇attention!
decode_attention!
```

## Softmax

```@docs
softmax
softmax!
∇softmax
∇softmax!
```

## Normalization

```@docs
rms_norm
rms_norm!
∇rms_norm
∇rms_norm!
layer_norm
layer_norm!
∇layer_norm
∇layer_norm!
```

## FlexAttention

```@docs
flex_attention
flex_attention!
∇flex_attention
∇flex_attention!
```

### Mask mods

```@docs
FullMask
CausalMask
SlidingWindowMask
PrefixMask
DocumentMask
AndMask
OrMask
prefix_lm
```

### Score mods

```@docs
NoOpScore
SoftCapScore
AliBiScore
BiasScore
ComposeScore
```

### Pair features

```@docs
PairFeatureScore
pair_feature
∇pair_feature
```

### Score mod gradients

```@docs
grad_shadow
```

### Block sparsity

```@docs
BlockMask
build_block_mask
```

### Host evaluation

```@docs
hmask
hscore
```
