import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(43)

# Definir el bloque residual
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(features, 32)
        self.layer2 = nn.Linear(32, 32)
        # Asegurarnos de que ReLU no se aplique in-place
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x  # Guardamos la entrada original para la conexión residual
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        
        # Añadimos la conexión residual sin modificar in-place
        out = out + identity  # Evitar operación in-place
        return out

# Red Neuronal Profunda con bloques residuales (ajustado a 2 bloques)
class DeepResidualNetwork(nn.Module):
    def __init__(self, input_features):
        super(DeepResidualNetwork, self).__init__()
        self.input_layer = nn.Linear(input_features, 32)
        self.activation = nn.ReLU(inplace=False)
        
        # Reducido a 2 bloques residuales
        self.residual_block1 = ResidualBlock(32)
        self.residual_block2 = ResidualBlock(32)

        # Capa de salida
        self.output_layer = nn.Linear(32, 1)  # Para clasificación binaria
        self.dropout = nn.Dropout(0.6)  # Ajustar el dropout a 0.5

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        
        # Pasar a través de los bloques residuales
        x = self.dropout(self.residual_block1(x))
        x = self.dropout(self.residual_block2(x))

        # Capa de salida
        x = self.output_layer(x)
        return x

# Función de entrenamiento
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

        # Convertimos los logits en probabilidades aplicando Sigmoid para la evaluación
        pred = torch.sigmoid(pred)  # Aplica sigmoid para obtener las probabilidades
        # Si pred contiene probabilidades, usamos 0.5 como umbral para clase binaria
        pred_class = (pred > 0.5).type(torch.float)  # Clase 1 si probabilidad > 0.5
        correct += (pred_class == y).type(torch.float).sum().item()

    # Promedio de pérdida y precisión
    train_loss /= len(dataloader)
    correct /= size
    return train_loss, correct

# Función de validación/prueba
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

            # Convertimos los logits en probabilidades aplicando Sigmoid para la evaluación
            pred = torch.sigmoid(pred)  # Aplica sigmoid para obtener las probabilidades
            # Umbral de 0.5 para clasificación binaria
            pred_class = (pred > 0.5).type(torch.float)
            correct += (pred_class == y).type(torch.float).sum().item()

    # Promedio de pérdida y precisión
    test_loss /= num_batches
    correct /= size
    return test_loss, correct


if __name__ == "__main__":
    # Cargar el dataset y preprocesar
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

    batch_size = 32
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Crear la red neuronal con 2 bloques residuales
    model = DeepResidualNetwork(X_train_tensor.shape[1])

    # Definir la función de pérdida (entropía cruzada binaria) y el optimizador (SGD con L2 regularization)
    criterion = nn.BCEWithLogitsLoss()  # Para clasificación binaria
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-8)  # Reducir la tasa de aprendizaje y añadir L2

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
