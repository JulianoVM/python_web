from machine_learning_proj import app
from machine_learning_proj import clf_dt, clf_knn, clf_mlp, clf_rf
from flask import render_template, request

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classifier', methods=['POST'])
def classifier():
    #recebe, com o "request", o classificador selecionado

    #executa o código dependendo do classificador selecionado e retorna ela para as variáveis "accuracy" e "f1"

    return render_template('index.html', accuracy=accuracy, f1=f1)