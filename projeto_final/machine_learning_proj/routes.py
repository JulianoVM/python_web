from machine_learning_proj import app
from machine_learning_proj.clf_mlp import mlp
from machine_learning_proj.clf_dt import dt
from machine_learning_proj.clf_rf import rf
from machine_learning_proj.clf_knn import knn
from flask import render_template, request, redirect, url_for

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():
    selected_classifier = request.form['classificadores']

    return redirect(url_for('pera', classifier=selected_classifier))


@app.route('/pera_ai/<classifier>')
def pera(classifier):
    return render_template(f'{classifier}.html')


@app.route('/result', methods=['POST'])
def result():
    classifier = request.form['classificadores']
    classifier = classifier.lower()

    if (classifier == 'knn'):
        parameter1 = request.form.get('n_neighbors', type=int, default=112)
        parameter2 = request.form.get('leaf_size', type=int, default=50)

        accuracy, f1 = knn(parameter1, parameter2)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)

    if (classifier == 'mlp'):
        parameter1 = request.form.get('learning_rate_init', type=float, default=0.001)
        parameter2 = request.form.get('max_iter', type=int, default=500)
        parameter3 = request.form.get('random_state', type=int, default=42)

        accuracy, f1 = mlp(parameter1, parameter2, parameter3)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)

    if (classifier == 'dt'):
        parameter1 = request.form.get('max_depth', type=int, default=100)
        parameter2 = request.form.get('max_features', type=float, default=0.5)
        parameter3 = request.form.get('random_state', type=int, default=42)

        accuracy, f1 = dt(parameter1, parameter2, parameter3)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)

    if (classifier == 'rf'):
        parameter1 = request.form.get('n_estimators', type=int, default=100)
        parameter2 = request.form.get('max_leaf_nodes', type=int, default=2)
        parameter3 = request.form.get('random_state', type=int, default=42)

        accuracy, f1 = rf(parameter1, parameter2, parameter3)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)
