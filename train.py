"""Train the fr -> Guyanais Creole GRU attention model."""
import csv
import json
import os
import sys
import time

import tensorflow as tf

from model import build_models
from text_utils import (
    standardize,
    standardize_target,
    build_vectorizer,
    save_vocab,
)

TRAIN_CSV = "data_train.csv"
VAL_CSV = "data_val.csv"
ARTIFACT_DIR = "artifacts"
SEQ_LEN = 12
EMBEDDING_DIM = 128
UNITS = 256
BATCH_SIZE = 32
EPOCHS = 120


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["fr"] for r in rows], [r["cr"] for r in rows]


def make_dataset(fr_texts, cr_texts, src_vec, tgt_vec, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((fr_texts, cr_texts))
    if shuffle:
        ds = ds.shuffle(len(fr_texts), seed=42)
    ds = ds.batch(BATCH_SIZE)

    def _vectorize(fr, cr):
        return src_vec(fr), tgt_vec(cr)

    return ds.map(_vectorize, num_parallel_calls=tf.data.AUTOTUNE)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def masked_loss(labels, logits):
    mask = tf.cast(labels != 0, tf.float32)
    loss = loss_object(labels, logits) * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def masked_accuracy(labels, logits):
    preds = tf.argmax(logits, axis=-1, output_type=labels.dtype)
    mask = tf.cast(labels != 0, tf.float32)
    correct = tf.cast(preds == labels, tf.float32) * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)


def train_step(encoder, decoder, optimizer, src, tgt):
    with tf.GradientTape() as tape:
        enc_out, state, enc_mask = encoder(src)
        loss = 0.0
        acc = 0.0
        steps = tgt.shape[1] - 1
        for t in range(steps):
            dec_input = tgt[:, t : t + 1]
            label = tgt[:, t + 1]
            logits, state, _ = decoder(dec_input, state, enc_out, enc_mask)
            loss += masked_loss(label, logits)
            acc += masked_accuracy(label, logits)
        loss /= steps
        acc /= steps

    variables = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss, acc


def eval_step(encoder, decoder, src, tgt):
    enc_out, state, enc_mask = encoder(src)
    loss = 0.0
    acc = 0.0
    steps = tgt.shape[1] - 1
    for t in range(steps):
        dec_input = tgt[:, t : t + 1]
        label = tgt[:, t + 1]
        logits, state, _ = decoder(dec_input, state, enc_out, enc_mask)
        loss += masked_loss(label, logits)
        acc += masked_accuracy(label, logits)
    return loss / steps, acc / steps


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    fr_train, cr_train = read_csv(TRAIN_CSV)
    fr_val, cr_val = read_csv(VAL_CSV)

    print(f"Train pairs: {len(fr_train)}  Val pairs: {len(fr_val)}")

    src_vec = build_vectorizer(fr_train, max_tokens=4000, standardize_fn=standardize, seq_len=SEQ_LEN)
    tgt_vec = build_vectorizer(
        cr_train, max_tokens=4000, standardize_fn=standardize_target, seq_len=SEQ_LEN
    )

    src_vocab_size = src_vec.vocabulary_size()
    tgt_vocab_size = tgt_vec.vocabulary_size()
    print(f"Source vocab: {src_vocab_size}  Target vocab: {tgt_vocab_size}")

    save_vocab(src_vec, os.path.join(ARTIFACT_DIR, "src_vocab.json"))
    save_vocab(tgt_vec, os.path.join(ARTIFACT_DIR, "tgt_vocab.json"))
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
        json.dump(
            {
                "seq_len": SEQ_LEN,
                "embedding_dim": EMBEDDING_DIM,
                "units": UNITS,
                "src_vocab_size": src_vocab_size,
                "tgt_vocab_size": tgt_vocab_size,
            },
            f,
        )

    train_ds = make_dataset(fr_train, cr_train, src_vec, tgt_vec, shuffle=True)
    val_ds = make_dataset(fr_val, cr_val, src_vec, tgt_vec, shuffle=False)

    encoder, decoder = build_models(src_vocab_size, tgt_vocab_size, EMBEDDING_DIM, UNITS)
    optimizer = tf.keras.optimizers.Adam()

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc, n = 0.0, 0.0, 0
        for src, tgt in train_ds:
            loss, acc = train_step(encoder, decoder, optimizer, src, tgt)
            train_loss += loss
            train_acc += acc
            n += 1
        train_loss /= n
        train_acc /= n

        val_loss, val_acc, nv = 0.0, 0.0, 0
        for src, tgt in val_ds:
            loss, acc = eval_step(encoder, decoder, src, tgt)
            val_loss += loss
            val_acc += acc
            nv += 1
        val_loss /= nv
        val_acc /= nv

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
                f"({time.time() - start:.1f}s)"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            encoder.save_weights(os.path.join(ARTIFACT_DIR, "encoder.weights.h5"))
            decoder.save_weights(os.path.join(ARTIFACT_DIR, "decoder.weights.h5"))

    print(f"Done. Best val_loss={best_val_loss:.4f}. Artifacts saved to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
