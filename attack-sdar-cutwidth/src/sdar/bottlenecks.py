import numpy as np
import tensorflow as tf
from keras import layers


def _activation_dim(input_shape):
    dim = 1
    for size in input_shape:
        dim *= int(size)
    return dim


def _resolve_k(input_shape, compression_ratio, bottleneck_k):
    dim = _activation_dim(input_shape)
    if bottleneck_k is not None and int(bottleneck_k) > 0:
        k = int(bottleneck_k)
    else:
        if compression_ratio <= 0.0:
            raise ValueError("compression_ratio must be positive.")
        k = int(round(dim / compression_ratio))
    return max(1, min(dim, k))


def dp_clip_and_noise(payload, clip_norm, sigma):
    """Per-sample L2 clip + Gaussian noise on the transmitted payload.

    TF2 mirror of the utility harness ``SplitModel._apply_dp`` (train_split.py:543-554):
    clip each sample's payload to ``clip_norm`` in L2, then add Gaussian noise with
    std = ``sigma * clip_norm``. Gated byte-identical when ``clip_norm <= 0`` so the
    no-DP path is untouched. Applied inside the bottleneck ``call`` (before the lift
    back) so the noised payload propagates to the lifted ``z`` that the server and the
    SDAR attacker observe, and to the eval-time SSIM.
    """
    if clip_norm is None or clip_norm <= 0:
        return payload
    orig_shape = tf.shape(payload)
    flat = tf.reshape(payload, [orig_shape[0], -1])
    norms = tf.norm(flat, axis=1, keepdims=True)
    clip_factor = tf.minimum(1.0, clip_norm / (norms + 1e-8))
    flat = flat * clip_factor
    if sigma is not None and sigma > 0:
        flat = flat + tf.random.normal(tf.shape(flat), dtype=flat.dtype) * (sigma * clip_norm)
    return tf.reshape(flat, orig_shape)


class IdentityBottleneck(layers.Layer):
    def __init__(self, input_shape, dp_clip=0.0, dp_sigma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(int(v) for v in input_shape)
        self.payload_dim = _activation_dim(self.input_shape_)
        self.effective_compression_ratio = 1.0
        self.dp_clip = float(dp_clip)
        self.dp_sigma = float(dp_sigma)

    def call(self, z, training=False):
        del training
        payload = tf.reshape(z, [tf.shape(z)[0], -1])
        payload = dp_clip_and_noise(payload, self.dp_clip, self.dp_sigma)
        # the "raw" smashed data IS the (noised) payload; reshape back so the server
        # and attacker observe the DP-perturbed activation.
        return tf.reshape(payload, tf.shape(z)), payload


class FixedRandomBottleneck(layers.Layer):
    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=0,
                 dp_clip=0.0, dp_sigma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(int(v) for v in input_shape)
        self.activation_dim = _activation_dim(self.input_shape_)
        self.k = _resolve_k(self.input_shape_, compression_ratio, bottleneck_k)
        rand_matrix = np.random.randn(self.activation_dim, self.k).astype("float32")
        q, _ = np.linalg.qr(rand_matrix)
        self.random_basis = tf.constant(q[:, :self.k], dtype=tf.float32)
        self.payload_dim = self.k
        self.effective_compression_ratio = self.activation_dim / self.payload_dim
        self.dp_clip = float(dp_clip)
        self.dp_sigma = float(dp_sigma)

    def call(self, z, training=False):
        del training
        flat = tf.reshape(z, [tf.shape(z)[0], self.activation_dim])
        basis = tf.cast(self.random_basis, flat.dtype)
        payload = tf.matmul(flat, basis)
        # DP on the transmitted k-dim payload BEFORE lift-back, so the lifted ĥ that
        # the server/attacker reconstructs from carries the clip+noise.
        payload = dp_clip_and_noise(payload, self.dp_clip, self.dp_sigma)
        lifted = tf.matmul(payload, basis, transpose_b=True)
        return tf.reshape(lifted, tf.shape(z)), payload




class TopKSparseBottleneck(layers.Layer):
    """Magnitude-based top-k sparsification (matched-byte compression baseline).

    Mirrors the deployed ``TopKSparseBottleneck``: keep the ``k_s``
    largest-magnitude coordinates per sample and zero the rest. The lifted map sent
    to the server is the dense d-dim *sparse* activation (kept coords, rest 0), so the
    SDAR attacker sees exactly what the real server sees. Matched-byte budget: keeping
    ``k_s`` coords costs ``k_s * (VALUE_BYTES + INDEX_BYTES)`` bytes, set equal to the
    LightSplit budget of ``k`` float32 values, i.e. ``k_s = floor(4k / 6)``.

    This is the true magnitude sparsifier from the utility table, so its SDAR
    SSIM is comparable to the matched-byte utility numbers.
    """

    VALUE_BYTES = 4
    INDEX_BYTES = 2

    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=0, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(int(v) for v in input_shape)
        self.activation_dim = _activation_dim(self.input_shape_)
        self.k = _resolve_k(self.input_shape_, compression_ratio, bottleneck_k)
        self.k_s = max(1, (self.VALUE_BYTES * self.k) //
                       (self.VALUE_BYTES + self.INDEX_BYTES))
        self.k_s = min(self.k_s, self.activation_dim)
        # payload is the k_s kept values; the byte budget equals k float32 values, so
        # the effective (byte-matched) compression ratio is reported against k.
        self.payload_dim = self.k_s
        self.effective_compression_ratio = self.activation_dim / self.k

    def call(self, z, training=False):
        del training
        batch_size = tf.shape(z)[0]
        flat = tf.reshape(z, [batch_size, self.activation_dim])
        _, idx = tf.math.top_k(tf.abs(flat), k=self.k_s, sorted=False)  # [B, k_s]
        b_idx = tf.repeat(tf.range(batch_size), self.k_s)
        scatter_idx = tf.stack([b_idx, tf.reshape(idx, [-1])], axis=1)
        mask = tf.scatter_nd(
            scatter_idx,
            tf.ones([batch_size * self.k_s], dtype=flat.dtype),
            tf.stack([batch_size, self.activation_dim]),
        )
        mask = tf.minimum(mask, tf.ones_like(mask))  # dedupe ties to 0/1
        lifted = flat * mask
        payload = tf.gather(flat, idx, batch_dims=1)  # kept values, for reporting/WCC
        return tf.reshape(lifted, tf.shape(z)), payload


class Learned1x1Bottleneck(layers.Layer):
    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=0, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(int(v) for v in input_shape)
        if len(self.input_shape_) != 3:
            raise ValueError("learned_1x1 bottleneck requires a rank-3 spatial activation shape.")
        height, width, channels = self.input_shape_
        self.activation_dim = _activation_dim(self.input_shape_)
        flat_k = _resolve_k(self.input_shape_, compression_ratio, bottleneck_k)
        spatial_dim = height * width
        self.k_channels = max(1, min(channels, int(round(flat_k / spatial_dim))))
        self.compress = layers.Conv2D(self.k_channels, 1, name="learned_1x1_compress")
        self.decompress = layers.Conv2D(channels, 1, name="learned_1x1_decompress")
        self.payload_dim = self.k_channels * spatial_dim
        self.effective_compression_ratio = self.activation_dim / self.payload_dim

    def call(self, z, training=False):
        payload = self.compress(z, training=training)
        lifted = self.decompress(payload, training=training)
        return lifted, payload


class BlockRandomBottleneck(layers.Layer):
    """Block-local fixed orthonormal random projection: a privacy/utility dial.

    Tiles the (H, W, C) cut into block x block spatial tiles, flattens each tile to
    C*block*block dims, and projects it with ONE shared fixed orthonormal random R
    to out_per_block dims.  The macro grid (H/block x W/block) is preserved, so the
    random mixing is confined to a block x block receptive field -- the knob between
    privacy and utility:
        block = 1   -> per-position channel projection (max utility, min privacy)
        block = H   -> single global dense R == FixedRandomBottleneck (max privacy)
    R is a client-only constant (never trained, never transmitted).  At a fixed
    compression ratio the transmitted payload (n_blocks * out_per_block floats) is
    held constant across block sizes, so block is the only moving variable.

    TF port of the PyTorch BlockRandomBottleneck (the utility harness): tiling is done with
    tf.nn.space_to_depth / depth_to_space (exact inverses), so the within-tile element
    ordering is internally consistent forward and backward; since R is random the
    ordering convention is irrelevant to the method.
    """
    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=0, block_size=2, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(int(v) for v in input_shape)
        if len(self.input_shape_) != 3:
            raise ValueError("block_random bottleneck requires a rank-3 spatial activation shape.")
        height, width, channels = self.input_shape_  # channels-last (H, W, C)
        self.block = int(block_size)
        if self.block < 1 or height % self.block != 0 or width % self.block != 0:
            raise ValueError(
                f"block_size={self.block} must be >=1 and divide spatial dims ({height},{width})."
            )
        self.height, self.width, self.channels = height, width, channels
        self.grid_h, self.grid_w = height // self.block, width // self.block
        self.n_blocks = self.grid_h * self.grid_w
        self.in_per_block = channels * self.block * self.block
        self.activation_dim = _activation_dim(self.input_shape_)
        k = _resolve_k(self.input_shape_, compression_ratio, bottleneck_k)
        # floor-divide the float budget across tiles (exact for square 2^n grids).
        self.out_per_block = max(1, min(self.in_per_block, k // self.n_blocks))
        rand_matrix = np.random.randn(self.in_per_block, self.out_per_block).astype("float32")
        q, _ = np.linalg.qr(rand_matrix)  # (in_per_block, out_per_block), orthonormal cols
        self.random_basis = tf.constant(q[:, :self.out_per_block], dtype=tf.float32)
        self.payload_dim = self.n_blocks * self.out_per_block
        self.effective_compression_ratio = self.activation_dim / self.payload_dim

    def call(self, z, training=False):
        del training
        batch = tf.shape(z)[0]
        # (B,H,W,C) -> (B, grid_h, grid_w, C*block*block): each block x block x C tile
        # is packed into the channel dim; depth_to_space is its exact inverse.  block=1
        # is per-position (no tiling); space_to_depth requires block>=2 so we skip it.
        tiles = tf.nn.space_to_depth(z, self.block) if self.block > 1 else z
        flat = tf.reshape(tiles, [-1, self.in_per_block])
        basis = tf.cast(self.random_basis, flat.dtype)
        proj = tf.matmul(flat, basis)                            # block-local projection
        lifted_flat = tf.matmul(proj, basis, transpose_b=True)   # lift back with R^T
        lifted_tiles = tf.reshape(lifted_flat, [batch, self.grid_h, self.grid_w, self.in_per_block])
        if self.block > 1:
            lifted = tf.nn.depth_to_space(lifted_tiles, self.block)
        else:
            lifted = tf.reshape(lifted_tiles, [batch, self.height, self.width, self.channels])
        payload = tf.reshape(proj, [batch, self.payload_dim])
        return lifted, payload


def make_bottleneck(mode, input_shape, compression_ratio=1.0, bottleneck_k=0, block_size=2,
                    dp_clip=0.0, dp_sigma=0.0):
    mode = (mode or "none").lower()
    if mode == "none":
        return IdentityBottleneck(
            input_shape, dp_clip=dp_clip, dp_sigma=dp_sigma, name="identity_bottleneck")
    if mode == "fixed_random":
        return FixedRandomBottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            dp_clip=dp_clip,
            dp_sigma=dp_sigma,
            name="fixed_random_bottleneck",
        )
    if mode == "topk_sparse":
        return TopKSparseBottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            name="topk_sparse_bottleneck",
        )
    if mode == "learned_1x1":
        return Learned1x1Bottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            name="learned_1x1_bottleneck",
        )
    if mode == "block_random":
        return BlockRandomBottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            block_size=block_size,
            name="block_random_bottleneck",
        )
    raise ValueError(f"Unknown bottleneck_mode: {mode}")


def within_class_compaction_loss(payload, labels):
    payload = tf.reshape(payload, [tf.shape(payload)[0], -1])
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    _, segment_ids = tf.unique(labels)
    num_segments = tf.reduce_max(segment_ids) + 1
    counts = tf.math.unsorted_segment_sum(
        tf.ones([tf.shape(payload)[0], 1], dtype=payload.dtype),
        segment_ids,
        num_segments,
    )
    sums = tf.math.unsorted_segment_sum(payload, segment_ids, num_segments)
    means = sums / tf.maximum(counts, tf.ones_like(counts))
    centered = payload - tf.gather(means, segment_ids)
    sq_dist = tf.reduce_sum(tf.square(centered), axis=1)
    class_loss = tf.math.unsorted_segment_sum(sq_dist, segment_ids, num_segments)
    class_loss = class_loss / tf.squeeze(tf.maximum(counts, tf.ones_like(counts)), axis=1)
    valid = tf.squeeze(counts, axis=1) > 1.0
    return tf.reduce_sum(tf.where(valid, class_loss, tf.zeros_like(class_loss)))


def center_loss_value(payload, labels, centers):
    """Center loss (Wen et al. 2016) value on the payload.

    Same reduction as ``within_class_compaction_loss`` (sum over present classes of the
    per-class mean squared distance), but the target is a PERSISTENT, EMA-updated center
    (the ``centers`` Variable) instead of the per-batch class mean -- the one
    methodological difference being studied in the WCC-vs-center-loss comparison.
    Persistent centers give a valid target even for singleton classes, so (unlike WCC)
    we do not skip them. Gradient flows only to ``payload``; centers are stop-gradient
    here and EMA-updated separately by ``update_centers`` after the optimizer step.
    """
    payload = tf.reshape(payload, [tf.shape(payload)[0], -1])
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    ctr = tf.stop_gradient(tf.gather(centers, labels))            # [B, k]
    sq_dist = tf.reduce_sum(tf.square(payload - ctr), axis=1)     # [B]
    _, segment_ids = tf.unique(labels)
    num_segments = tf.reduce_max(segment_ids) + 1
    counts = tf.math.unsorted_segment_sum(tf.ones_like(sq_dist), segment_ids, num_segments)
    class_loss = tf.math.unsorted_segment_sum(sq_dist, segment_ids, num_segments)
    class_loss = class_loss / tf.maximum(counts, tf.ones_like(counts))
    return tf.reduce_sum(class_loss)


def update_centers(payload, labels, centers, alpha):
    """EMA update of persistent class centers toward this batch's per-class means
    (Wen et al. 2016 rule); only classes present in the batch are updated. Mutates the
    ``centers`` tf.Variable in place; carries no gradient, so call it OUTSIDE the
    gradient tape, after the optimizer step.
    """
    payload = tf.reshape(payload, [tf.shape(payload)[0], -1])
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    num_classes = tf.shape(centers)[0]
    counts = tf.math.unsorted_segment_sum(
        tf.ones([tf.shape(payload)[0], 1], dtype=payload.dtype), labels, num_classes)   # [C,1]
    sums = tf.math.unsorted_segment_sum(payload, labels, num_classes)                    # [C,k]
    means = sums / tf.maximum(counts, tf.ones_like(counts))                              # [C,k]
    present = tf.broadcast_to(counts > 0.0, tf.shape(centers))                           # [C,k]
    new_centers = centers + alpha * (means - centers)
    centers.assign(tf.where(present, new_centers, centers))
