"""Flask web app for the fr -> Guyanais Creole translator."""
from flask import Flask, jsonify, render_template, request

from translate import load_everything, translate

app = Flask(__name__)

print("Loading model...")
encoder, decoder, src_vec, tgt_vocab_arr, config = load_everything()
print("Model loaded.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a French sentence."}), 400
    if len(text) > 300:
        return jsonify({"error": "Sentence is too long."}), 400

    result = translate(text, encoder, decoder, src_vec, tgt_vocab_arr, config)
    return jsonify({"translation": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
