from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/nlplab", methods=["POST"])
def nlplab():
    response = {"body":"test"}
    return jsonify(response)