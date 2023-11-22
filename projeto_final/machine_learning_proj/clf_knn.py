import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

def knn(num_n_neighbors, num_leaf_size):
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = KNeighborsClassifier(n_neighbors=num_n_neighbors, leaf_size=num_leaf_size)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    classes = iris.target_names.tolist()
    display_mlp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    display_mlp.plot()
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Previsão')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão knn')
    plt.savefig('C:/Users/guias/OneDrive/Área de Trabalho/piton/python_web/projeto_final/machine_learning_proj/images/matriz_confusao.png')
    plt.close()

    return accuracy, f1