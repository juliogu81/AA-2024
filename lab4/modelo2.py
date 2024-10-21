import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Definir el modelo con una sola neurona lineal y dos salidas (para 2 clases)
class SimpleNN(nn.Module):
    def __init__(self, features):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(features, 2)  #features es el número de características

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    # Cargar el dataset con encabezados y eliminar la columna 'time'
    DATASET_FILE = 'lab4_dataset.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=",")
    dataset = dataset.drop(columns=['time'])
    dataset = dataset.drop(columns=['pidnum'])

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

    # Convertir los datos a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

   # Crear la red neuronal
    model = SimpleNN(X_train_tensor.shape[1])

    # Definir la función de pérdida (entropía cruzada) y el optimizador (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Entrenar la red durante 100 épocas
    num_epochs = 100
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Modo de entrenamiento
        model.train()
        optimizer.zero_grad()  # Resetear gradientes
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Calcular pérdida
        loss.backward()  # Backward pass
        optimizer.step()  # Actualizar parámetros

        # Almacenar la pérdida para entrenamiento
        train_losses.append(loss.item())

        # Calcular la precisión (accuracy) en el conjunto de entrenamiento
        _, predicted = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train_tensor, predicted)
        train_accuracies.append(train_acc)

        # Evaluar en el conjunto de validación (sin cálculo de gradientes)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())

            # Calcular la precisión (accuracy) en el conjunto de validación
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_val_tensor, val_predicted)
            val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')

    # Graficar la pérdida y la accuracy a lo largo de las épocas
    plt.figure(figsize=(12, 5))

    # Gráfico de la pérdida
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Entrenamiento')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    # Gráfico de la accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Entrenamiento')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()