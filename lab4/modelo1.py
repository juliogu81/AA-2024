import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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