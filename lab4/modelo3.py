import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



# Definir el modelo con una neurona, activación sigmoide y una salida
class SimpleSigmoidNN(nn.Module):
    def __init__(self, features):
        super(SimpleSigmoidNN, self).__init__()
        self.linear = nn.Linear(features, 1)  # Una neurona, una salida
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Activación sigmoide


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # Modo de entrenamiento
    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        # Forward pass
        pred = model(X).squeeze()  # Aplanar la salida para que sea un vector
        loss = loss_fn(pred, y)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcular métricas
        train_loss += loss.item()

        # Si pred contiene probabilidades, usamos 0.5 como umbral para clase binaria
        pred_class = (pred > 0.5).type(torch.float)  # Clase 1 si probabilidad > 0.5
        correct += (pred_class == y).type(torch.float).sum().item()

    # Promedio de pérdida y precisión
    train_loss /= len(dataloader)
    correct /= size
    return train_loss, correct


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Modo de evaluación
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).squeeze()  # Aplanar la salida para que sea un vector
            test_loss += loss_fn(pred, y).item()

            # Umbral de 0.5 para clasificación binaria
            pred_class = (pred > 0.5).type(torch.float)
            correct += (pred_class == y).type(torch.float).sum().item()

    # Promedio de pérdida y precisión
    test_loss /= num_batches
    correct /= size
    return test_loss, correct



if __name__ == "__main__":
    # Cargar el dataset y preprocesar (igual que antes)
    DATASET_FILE = 'lab4_dataset.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=",")
    dataset = dataset.drop(columns=['time', 'pidnum'])

    # Dividir el conjunto de datos
    train_full, test = train_test_split(dataset, test_size=0.2, random_state=42)
    train, validation = train_test_split(train_full, test_size=0.1, random_state=42)

    X_train = train.drop(columns=['cid'])
    y_train = train['cid']
    X_val = validation.drop(columns=['cid'])
    y_val = validation['cid']

    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convertir a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 10
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Crear la red neuronal
    model = SimpleSigmoidNN(X_train_tensor.shape[1])

    # Definir la función de pérdida (entropía cruzada binaria) y el optimizador (SGD)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Entrenamiento
    num_epochs = 100
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Ejecutar el ciclo de entrenamiento
        train_loss, train_acc = train_loop(dataloader_train, model, criterion, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Ejecutar el ciclo de validación
        val_loss, val_acc = test_loop(dataloader_val, model, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    # Graficar las pérdidas y accuracies
    plt.figure(figsize=(12, 5))

    # Gráfico de la pérdida
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Entrenamiento')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validación')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    # Gráfico de la accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Entrenamiento')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validación')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()