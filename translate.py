"""Translate French sentences to Guyanais Creole with the trained model.

Usage:
    python translate.py "Comment tu vas ?"
    python translate.py            # interactive prompt
"""
import json
import os
import sys

import tensorflow as tf

from model import build_models
from text_utils import (
    standardize,
    standardize_target,
    load_vocab,
    vectorizer_from_vocab,
    START_TOK,
    END_TOK,
)

ARTIFACT_DIR = "artifacts"


def load_everything():
    with open(os.path.join(ARTIFACT_DIR, "config.json")) as f:
        config = json.load(f)

    src_vocab = load_vocab(os.path.join(ARTIFACT_DIR, "src_vocab.json"))
    tgt_vocab = load_vocab(os.path.join(ARTIFACT_DIR, "tgt_vocab.json"))

    src_vec = vectorizer_from_vocab(src_vocab, standardize, seq_len=config["seq_len"])
    tgt_vec = vectorizer_from_vocab(tgt_vocab, standardize_target, seq_len=None)

    encoder, decoder = build_models(
        config["src_vocab_size"], config["tgt_vocab_size"],
        config["embedding_dim"], config["units"],
    )
    # Build variables with a dummy forward pass, then load trained weights.
    dummy = tf.zeros((1, config["seq_len"]), dtype=tf.int64)
    enc_out, state, enc_mask = encoder(dummy)
    decoder(dummy[:, :1], state, enc_out, enc_mask)

    encoder.load_weights(os.path.join(ARTIFACT_DIR, "encoder.weights.h5"))
    decoder.load_weights(os.path.join(ARTIFACT_DIR, "decoder.weights.h5"))

    tgt_vocab_arr = tgt_vec.get_vocabulary()

    return encoder, decoder, src_vec, tgt_vocab_arr, config


def translate(sentence, encoder, decoder, src_vec, tgt_vocab_arr, config, max_len=20):
    src = src_vec(tf.constant([sentence]))
    enc_out, state, enc_mask = encoder(src)

    start_id = tgt_vocab_arr.index(START_TOK)
    end_id = tgt_vocab_arr.index(END_TOK)

    dec_input = tf.constant([[start_id]], dtype=tf.int64)
    result = []
    for _ in range(max_len):
        logits, state, _ = decoder(dec_input, state, enc_out, enc_mask)
        token_id = int(tf.argmax(logits[0]).numpy())
        if token_id == end_id or token_id == 0:
            break
        result.append(tgt_vocab_arr[token_id])
        dec_input = tf.constant([[token_id]], dtype=tf.int64)

    return " ".join(result)


def main():
    encoder, decoder, src_vec, tgt_vocab_arr, config = load_everything()

    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        print(translate(sentence, encoder, decoder, src_vec, tgt_vocab_arr, config))
        return

    print("Type a French sentence (empty line to quit):")
    while True:
        try:
            sentence = input("fr> ").strip()
        except EOFError:
            break
        if not sentence:
            break
        print("cr>", translate(sentence, encoder, decoder, src_vec, tgt_vocab_arr, config))


if __name__ == "__main__":
    main()
