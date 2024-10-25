import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


if __name__ == "__main__":
    # Cargar el dataset con encabezados y eliminar la columna 'time'
    DATASET_FILE = 'lab4_dataset.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=",")
    dataset = dataset.drop(columns=['time'])

    print("Separando el conjunto de datos en conjunto de entrenamiento y conjunto de prueba...")
    # Dividir el conjunto de datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
    train_full, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Dividir el conjunto de entrenamiento completo en conjunto de entrenamiento (90%) y validación (10%)
    train, validation = train_test_split(train_full, test_size=0.1, random_state=42)

    # Separar características (X) y etiquetas (y) para el conjunto de entrenamiento y validación
    X_train = train.drop(columns=['cid'])  
    y_train = train['cid']

    X_val = validation.drop(columns=['cid'])
    y_val = validation['cid']

    print("Entrenando modelo de regresión logística...")

    # Crear el modelo de regresión logística
    model = LogisticRegression(max_iter=3000)

    # Entrenar el modelo con el conjunto de entrenamiento
    model.fit(X_train, y_train)

    # Predecir las etiquetas del conjunto de validación
    y_pred = model.predict(X_val)

    # Calcular la accuracy en el conjunto de validación
    accuracy = accuracy_score(y_val, y_pred)

    print(f'Accuracy del modelo de regresión logística: {accuracy:.4f}')

# Entrenamiento y predicción en el conjunto de prueba
    model2 = LogisticRegression(max_iter=3000)
    y_train_full = train_full['cid']
    x_train_full = train_full.drop(columns=['cid'])
    x_test = test.drop(columns=['cid'])
    y_test = test['cid']
    model2.fit(x_train_full, y_train_full)

    # Predecir las etiquetas del conjunto de prueba
    y_test_pred = model2.predict(x_test)

    # Calcular métricas en el conjunto de prueba
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)

    cm = confusion_matrix (y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.show()

    print(f'Accuracy del modelo de regresión logística (prueba): {accuracy_test:.4f}')
    print(f'Precision del modelo de regresión logística (prueba): {precision_test:.4f}')
    print(f'Recall del modelo de regresión logística (prueba): {recall_test:.4f}')
    print(f'F1 del modelo de regresión logística (prueba): {f1_test:.4f}')