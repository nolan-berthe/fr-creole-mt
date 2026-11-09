# French -> Guyanais Creole translator

A small TensorFlow seq2seq model (GRU encoder-decoder with Bahdanau attention)
trained on `fr_creole_guyanais_5000.xlsx`.

## About the data

The spreadsheet has 5200 rows, but only **520 unique French/Creole sentence
pairs** (each pair is duplicated exactly 10x). The 520 sentences themselves
are built from ~50 base phrases combined with a handful of prefixes/suffixes
("Alors, ...", "Est-ce que ...", "... maintenant", "... s'il te plaît", "...
!", "... ?"). `prepare_data.py` dedupes down to the 520 unique pairs and
splits them 90/10 into train/val.

**What this means in practice:** the model will reproduce these phrasebook
sentences (and close variants/recombinations) very well, but it has only seen
~50 real-world concepts — it will not generalize to arbitrary, unseen French
sentences the way a model trained on tens of thousands of diverse sentences
would. Treat it as a working proof-of-concept / demo, not a production
translator. To go further, the single biggest lever is more, more diverse
parallel sentences.

## Files

- `prepare_data.py` — reads the xlsx, dedupes, writes `data_pairs.csv` /
  `data_train.csv` / `data_val.csv`.
- `text_utils.py` — text standardization + vocabulary (TextVectorization)
  helpers.
- `model.py` — `Encoder` (bidirectional GRU), `BahdanauAttention`, `Decoder`.
- `train.py` — custom teacher-forcing training loop. Saves weights + vocab +
  config to `artifacts/`.
- `translate.py` — loads the trained model and greedily decodes French ->
  Creole, either from the command line or an interactive prompt.

## Usage

```bash
python prepare_data.py      # already run — regenerate if the xlsx changes
python train.py 120         # train for 120 epochs (default if no arg given)
python translate.py "Comment tu vas ?"
python translate.py         # interactive mode
```

## Notes

- Runs on CPU (TensorFlow 2.22.0rc0, the first version with a Windows/Python
  3.14 wheel at the time this was built).
- Sequence length is capped at 12 tokens (covers every sentence in the
  dataset after punctuation-splitting) to keep padding minimal, since this
  TF/Keras build has a bug where RNN masking on a `Bidirectional(GRU(...))`
  layer is silently dropped (cosmetic warning, doesn't affect correctness
  given the tight sequence length and the padding-aware loss/accuracy).
