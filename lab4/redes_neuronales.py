import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargar el dataset con encabezados y eliminar la columna 'time'
    DATASET_FILE = 'lab4_dataset.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=",")
    dataset = dataset.drop(columns=['time'])

    # Dividir el conjunto de datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
    train_full, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Dividir el conjunto de entrenamiento completo en conjunto de entrenamiento (90%) y validación (10%)
    train, validation = train_test_split(train_full, test_size=0.1, random_state=42)

    # Mostrar el tamaño de los conjuntos
    print(f"Conjunto de entrenamiento completo: {train_full.shape}")
    print(f"Conjunto de entrenamiento: {train.shape}")
    print(f"Conjunto de validación: {validation.shape}")
    print(f"Conjunto de prueba: {test.shape}")




   