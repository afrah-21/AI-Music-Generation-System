from flask import Flask, render_template, send_file
import subprocess
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate")
def generate_music():
    subprocess.run(["python", "generate.py"])

    return send_file(
        "generated_music.mid",
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)
    