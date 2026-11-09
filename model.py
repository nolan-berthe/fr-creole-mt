"""GRU encoder-decoder with Bahdanau attention for fr -> Creole translation."""
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        )
        self.combine_state = tf.keras.layers.Dense(units)

    def call(self, x):
        mask = tf.cast(x != 0, tf.bool)
        emb = self.embedding(x)
        seq, fwd_state, bwd_state = self.bigru(emb)
        state = self.combine_state(tf.concat([fwd_state, bwd_state], axis=-1))
        return seq, state, mask


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, query, values, mask):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query)))
        score = tf.squeeze(score, -1)
        score = tf.where(mask, score, tf.fill(tf.shape(score), float("-inf")))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(tf.expand_dims(weights, -1) * values, axis=1)
        return context, weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(units)
        self.fc1 = tf.keras.layers.Dense(units, activation="tanh")
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, state, enc_out, enc_mask):
        x = self.embedding(x)
        context, weights = self.attention(state, enc_out, enc_mask)
        x = tf.concat([x, tf.expand_dims(context, 1)], axis=-1)
        out, state = self.gru(x, initial_state=state)
        out = tf.squeeze(out, 1)
        out = self.fc1(tf.concat([out, context], axis=-1))
        logits = self.fc2(out)
        return logits, state, weights


def build_models(src_vocab_size, tgt_vocab_size, embedding_dim=128, units=256):
    encoder = Encoder(src_vocab_size, embedding_dim, units)
    decoder = Decoder(tgt_vocab_size, embedding_dim, units)
    return encoder, decoder
