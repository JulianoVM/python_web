import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support

def mlp(indice_inicial_aprendizado, num_max_iter, func_ativa):
    url = './static/csv/shopping_trends.csv'
    df_inicial = pd.read_csv(url)

    for column in df_inicial.columns:
        if df_inicial[column].dtype == 'object':
            df_inicial[column] = LabelEncoder().fit_transform(df_inicial[column])

    X = df_inicial.iloc[:, :-1]
    y = df_inicial.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    indice_inicial_aprendizado = 0.001
    num_max_iter = 100
    func_ativa = 'tanh'

    clf = MLPClassifier(hidden_layer_sizes=(100, 50, 25), learning_rate_init=indice_inicial_aprendizado, max_iter=num_max_iter, activation=func_ativa, random_state=42)
    clf.fit(X_train_scaled, y_train)

    predictions = clf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')

    return [accuracy, f1]

def dt():
    return 0

def rf():
    return 0

def knn():
    return 0