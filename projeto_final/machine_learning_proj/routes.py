from machine_learning_proj import app
from machine_learning_proj.clf_mlp import mlp
from machine_learning_proj.clf_dt import dt
from machine_learning_proj.clf_rf import rf
from machine_learning_proj.clf_knn import knn
from flask import render_template, request, redirect, url_for

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    selected_classifier = request.form['classificadores']

    return redirect(url_for('result', classifier=selected_classifier))

@app.route('/result/<classifier>', methods=['POST', 'GET'])
def result(classifier):
    classifier = classifier.lower()

    if (classifier == 'knn'):
        parameter1 = request.form.get('n_neighbors', type=int)
        parameter2 = request.form.get('leaf_size', type=int)

        accuracy, f1 = knn(parameter1, parameter2)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)

    if (classifier == 'mlp'):
        parameter1 = request.form.get('learning_rate_init', type=float)
        parameter2 = request.form.get('max_iter', type=int, default=1)
        parameter3 = request.form.get('random_state', type=int, default=0)

        accuracy, f1 = mlp(parameter1, parameter2, parameter3)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)

    if (classifier == 'dt'):
        parameter1 = request.form.get('max_depth', type=int)
        parameter2 = request.form.get('max_features', type=int)
        parameter3 = request.form.get('random_state', type=int)

        accuracy, f1 = dt(parameter1, parameter2, parameter3)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)

    if (classifier == 'rf'):
        parameter1 = request.form.get('n_estimators', type=int)
        parameter2 = request.form.get('max_leaf_nodes', type=int)
        parameter3 = request.form.get('random_state', type=int)

        accuracy, f1 = rf(parameter1, parameter2, parameter3)
        return render_template(f'{classifier}.html', classifier=classifier, accuracy=accuracy, f1=f1)
