from flask import Flask, request, render_template, jsonify

from usrlib.nlp import get_emocon_for_input, test_accuracy

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/nlplab", methods=["POST"])
def nlplab():
    text = request.json["text"]
    emocon = get_emocon_for_input(text)
    response = {"body": emocon}
    return jsonify(response)


@app.route("/test", methods=["GET"])
def nlplab_test():
    test = test_accuracy()
    return test