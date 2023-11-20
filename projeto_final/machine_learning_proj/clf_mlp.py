import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

def mlp(indice_inicial_aprendizado, num_max_iter, num_random_state):
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = MLPClassifier(learning_rate_init=indice_inicial_aprendizado, max_iter=num_max_iter, activation='tanh', random_state=num_random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    classes = iris.target_names.tolist()
    display_mlp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    display_mlp.plot()
    plt.figure(figsize=(6, 4))
    # sns.heatmap(matriz_confusao, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Previsão')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.savefig('caminho_especifico/matriz_confusao.png')  # Salvar a matriz de confusão em um caminho específico
    plt.close()

    return accuracy, f1, 