import numpy as np
import tensorflow as tf


def activation_dim(input_shape):
    dim = 1
    for size in input_shape:
        dim *= int(size)
    return dim


def _as_shape_tuple(input_shape):
    return tuple(int(size) for size in input_shape)


def _resolve_flat_k(input_shape, compression_ratio=1.0, bottleneck_k=None):
    dim = activation_dim(input_shape)
    if bottleneck_k is not None and int(bottleneck_k) > 0:
        k = int(bottleneck_k)
    else:
        compression_ratio = float(compression_ratio)
        if compression_ratio <= 0.0:
            raise ValueError("compression_ratio must be positive.")
        k = int(round(dim / compression_ratio))
    return max(1, min(dim, k))


def _resolve_channel_k(input_shape, compression_ratio=1.0, bottleneck_k=None):
    channels = int(input_shape[-1])
    if bottleneck_k is not None and int(bottleneck_k) > 0:
        k_channels = int(bottleneck_k)
    else:
        compression_ratio = float(compression_ratio)
        if compression_ratio <= 0.0:
            raise ValueError("compression_ratio must be positive.")
        k_channels = int(round(channels / compression_ratio))
    return max(1, min(channels, k_channels))


# Honest transmitted-byte accounting for sparse payloads (mirrors the utility harness):
# each kept coordinate costs a 4-byte float32 value + a 2-byte uint16 index.
VALUE_BYTES = 4
INDEX_BYTES = 2


class IdentityBottleneck(tf.keras.layers.Layer):
    supports_wcc = False

    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = _as_shape_tuple(input_shape)
        self.output_shape_ = self.input_shape_
        self.k = activation_dim(self.input_shape_)
        self.payload_dim = self.k
        self.effective_compression_ratio = 1.0

    def call(self, z, training=False):
        del training
        return z




class FixedRandomBottleneck(tf.keras.layers.Layer):
    supports_wcc = True

    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=None, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = _as_shape_tuple(input_shape)
        self.activation_dim = activation_dim(self.input_shape_)
        self.k = _resolve_flat_k(self.input_shape_, compression_ratio, bottleneck_k)
        self.output_shape_ = (self.k,)
        self.payload_dim = self.k
        self.effective_compression_ratio = self.activation_dim / self.payload_dim

        rand_matrix = np.random.randn(self.activation_dim, self.k).astype("float32")
        q, _ = np.linalg.qr(rand_matrix)
        self.projection = self.add_weight(
            name="projection",
            shape=(self.activation_dim, self.k),
            initializer=tf.keras.initializers.Constant(q[:, :self.k]),
            trainable=False,
        )

    def call(self, z, training=False):
        del training
        flat = tf.reshape(z, [tf.shape(z)[0], self.activation_dim])
        payload = tf.matmul(flat, tf.cast(self.projection, flat.dtype))
        payload.set_shape([None, self.k])
        return payload


class Learned1x1Bottleneck(tf.keras.layers.Layer):
    supports_wcc = False

    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=None, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = _as_shape_tuple(input_shape)
        if len(self.input_shape_) != 3:
            raise ValueError("learned_1x1 requires rank-3 spatial activations.")
        height, width, channels = self.input_shape_
        self.k_channels = _resolve_channel_k(self.input_shape_, compression_ratio, bottleneck_k)
        self.output_shape_ = (height, width, self.k_channels)
        self.k = self.k_channels
        self.payload_dim = height * width * self.k_channels
        self.effective_compression_ratio = activation_dim(self.input_shape_) / self.payload_dim
        self.compress = tf.keras.layers.Conv2D(
            self.k_channels,
            1,
            padding="same",
            name="learned_1x1_compress",
        )

    def call(self, z, training=False):
        return self.compress(z, training=training)


class TopKSparseBottleneck(tf.keras.layers.Layer):
    """Matched-byte magnitude sparsification (ports the utility harness TopKSparseBottleneck).

    Keeps the ``k_s`` largest-magnitude coordinates per sample and zeroes the rest. The
    payload stays dense (same spatial shape as the input) so the raw decoder/discriminator
    are reused unchanged; the honest transmitted cost is carried by ``payload_dim``.
    ``k_s`` is byte-matched against a dense ``k``-dim projection at the same CR:
    ``k_s * (VALUE_BYTES + INDEX_BYTES) ~= k * VALUE_BYTES``. Not WCC-compatible (WCC is a
    LightSplit component). Gradients flow straight through the kept coordinates (mask-multiply).
    """

    supports_wcc = False

    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=None, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = _as_shape_tuple(input_shape)
        self.output_shape_ = self.input_shape_          # dense payload, shape unchanged
        self.activation_dim = activation_dim(self.input_shape_)
        # nominal dense payload dim k = d / CR, then byte-matched sparsity budget k_s
        self.nominal_k = _resolve_flat_k(self.input_shape_, compression_ratio, bottleneck_k)
        self.k_s = max(1, (VALUE_BYTES * self.nominal_k) // (VALUE_BYTES + INDEX_BYTES))
        self.k = self.k_s
        # honest byte cost expressed as float32-equivalents: ceil(6 * k_s / 4)
        self.payload_dim = (self.k_s * (VALUE_BYTES + INDEX_BYTES) + VALUE_BYTES - 1) // VALUE_BYTES
        self.effective_compression_ratio = self.activation_dim / self.payload_dim

    def call(self, z, training=False):
        del training
        batch = tf.shape(z)[0]
        flat = tf.reshape(z, [batch, self.activation_dim])
        k_s = min(self.k_s, self.activation_dim)
        _, idx = tf.math.top_k(tf.abs(flat), k=k_s, sorted=False)        # (B, k_s) kept positions
        rows = tf.repeat(tf.range(batch), k_s)
        cols = tf.reshape(idx, [-1])
        mask = tf.scatter_nd(
            tf.stack([rows, cols], axis=1),
            tf.ones([batch * k_s], flat.dtype),
            [batch, self.activation_dim],
        )                                                                # (B, d) 0/1 keep-mask
        sparse = flat * mask
        return tf.reshape(sparse, tf.shape(z))


class BlockRandomBottleneck(tf.keras.layers.Layer):
    """Block-local fixed orthonormal random projection: a privacy/utility dial.

    Tiles the (H, W, C) cut into ``block x block`` spatial tiles, flattens each tile to
    ``C*block*block`` dims, and projects it with ONE shared fixed orthonormal random R to
    ``out_per_block`` dims. The transmitted payload is the concatenation of the per-block
    projections (``n_blocks * out_per_block`` floats); at a fixed compression ratio this is
    held constant across block sizes, so ``block`` is the only moving variable:
        block = 1   -> per-position channel projection (max utility, min privacy)
        block = H   -> single global dense R == FixedRandomBottleneck (max privacy)
    R is a client-only constant (never trained, never transmitted). Unlike the SDAR/MLP
    ports this emits the *payload* (no lift-back), matching FSHA's convention where the
    bottleneck output is what the decoder/discriminator consume. Tiling uses
    ``tf.nn.space_to_depth``; block=1 needs no tiling.
    """
    supports_wcc = True

    def __init__(self, input_shape, compression_ratio=1.0, bottleneck_k=None, block_size=2, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = _as_shape_tuple(input_shape)
        if len(self.input_shape_) != 3:
            raise ValueError("block_random bottleneck requires a rank-3 spatial activation shape.")
        height, width, channels = self.input_shape_
        self.block = int(block_size)
        if self.block < 1 or height % self.block != 0 or width % self.block != 0:
            raise ValueError(
                "block_size=%d must be >=1 and divide spatial dims (%d,%d)." % (self.block, height, width)
            )
        self.height, self.width, self.channels = height, width, channels
        self.grid_h, self.grid_w = height // self.block, width // self.block
        self.n_blocks = self.grid_h * self.grid_w
        self.in_per_block = channels * self.block * self.block
        self.activation_dim = activation_dim(self.input_shape_)
        k = _resolve_flat_k(self.input_shape_, compression_ratio, bottleneck_k)
        self.out_per_block = max(1, min(self.in_per_block, k // self.n_blocks))
        self.payload_dim = self.n_blocks * self.out_per_block
        self.k = self.payload_dim
        self.output_shape_ = (self.payload_dim,)
        self.effective_compression_ratio = self.activation_dim / self.payload_dim
        rand_matrix = np.random.randn(self.in_per_block, self.out_per_block).astype("float32")
        q, _ = np.linalg.qr(rand_matrix)
        self.projection = self.add_weight(
            name="block_projection",
            shape=(self.in_per_block, self.out_per_block),
            initializer=tf.keras.initializers.Constant(q[:, :self.out_per_block]),
            trainable=False,
        )

    def call(self, z, training=False):
        del training
        batch = tf.shape(z)[0]
        tiles = tf.nn.space_to_depth(z, self.block) if self.block > 1 else z
        flat = tf.reshape(tiles, [-1, self.in_per_block])
        payload = tf.matmul(flat, tf.cast(self.projection, flat.dtype))
        payload = tf.reshape(payload, [batch, self.payload_dim])
        payload.set_shape([None, self.payload_dim])
        return payload


def make_bottleneck(mode, input_shape, compression_ratio=1.0, bottleneck_k=None, block_size=2):
    mode = (mode or "none").lower()
    if mode in ("none", "raw", "identity"):
        return IdentityBottleneck(input_shape, name="identity_bottleneck")
    if mode == "fixed_random":
        return FixedRandomBottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            name="fixed_random_bottleneck",
        )
    if mode == "learned_1x1":
        return Learned1x1Bottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            name="learned_1x1_bottleneck",
        )
    if mode == "topk_sparse":
        return TopKSparseBottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            name="topk_sparse_bottleneck",
        )
    if mode == "block_random":
        return BlockRandomBottleneck(
            input_shape,
            compression_ratio=compression_ratio,
            bottleneck_k=bottleneck_k,
            block_size=block_size,
            name="block_random_bottleneck",
        )
    raise ValueError("Unknown bottleneck_mode: %s" % mode)


def within_class_compaction_loss(payload, labels):
    payload = tf.reshape(payload, [tf.shape(payload)[0], -1])
    labels = tf.convert_to_tensor(labels)
    if labels.shape.rank is not None and labels.shape.rank > 1 and labels.shape[-1] != 1:
        labels = tf.argmax(labels, axis=-1)
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
