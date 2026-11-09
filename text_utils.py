"""Shared text preprocessing for the fr -> Guyanais Creole translator."""
import json

import tensorflow as tf

START_TOK = "[start]"
END_TOK = "[end]"


def standardize(text):
    """Lowercase, normalize curly quotes, and space out sentence punctuation.

    Apostrophes and hyphens are left attached to their word (e.g. "qu'est-ce",
    "fenèt-la") since they are morphological, not separators, in both languages.
    """
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[’‘]", "'")
    text = tf.strings.regex_replace(text, r"([!?.,:;])", r" \1 ")
    text = tf.strings.regex_replace(text, r"\s+", " ")
    return tf.strings.strip(text)


def standardize_target(text):
    text = standardize(text)
    return tf.strings.join([START_TOK, text, END_TOK], separator=" ")


def build_vectorizer(texts, max_tokens, standardize_fn, seq_len):
    vec = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=standardize_fn,
        output_mode="int",
        output_sequence_length=seq_len,
    )
    vec.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(64))
    return vec


def save_vocab(vectorizer, path):
    vocab = vectorizer.get_vocabulary()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vectorizer_from_vocab(vocab, standardize_fn, seq_len=None):
    vec = tf.keras.layers.TextVectorization(
        standardize=standardize_fn,
        output_mode="int",
        vocabulary=vocab,
        output_sequence_length=seq_len,
    )
    return vec
