from machine_learning_proj import app
from machine_learning_proj import classifiers
from machine_learning_proj import plotagem
from flask import render_template

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classifier', methods=['POST'])
def classifier():
    return 0