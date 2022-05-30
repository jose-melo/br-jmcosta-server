from flask import Flask
from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/<name>")
def bla(name):
    return f"Hello, {escape(name)}"

@app.route("/testing")
def testing():
    return "<h1>Testing</h1>"