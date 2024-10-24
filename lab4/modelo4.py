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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.manual_seed(43)

class SigmoidNNWithHiddenLayer(nn.Module):
    def __init__(self, features):
        super(SigmoidNNWithHiddenLayer, self).__init__()
        self.hidden = nn.Linear(features, 16)   # Capa oculta con 16 unidades
        self.output = nn.Linear(16, 2)          # Capa de salida con 2 neuronas (para las dos clases)
    
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.softmax(self.output(x), dim=1) # Softmax para obtener probabilidades
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # Modo de entrenamiento
    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        # Forward pass
        pred = model(X)  # La salida ya tiene forma (batch_size, 2)
        loss = loss_fn(pred, y)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcular métricas
        train_loss += loss.item()

        # Obtener la clase predicha (índice de la mayor probabilidad)
        pred_class = pred.argmax(dim=1)  # Clase con la mayor probabilidad
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
            pred = model(X)  # La salida ya tiene forma (batch_size, 2)
            test_loss += loss_fn(pred, y).item()

            # Obtener la clase predicha (índice de la mayor probabilidad)
            pred_class = pred.argmax(dim=1)  # Clase con la mayor probabilidad
            correct += (pred_class == y).type(torch.float).sum().item()

    # Promedio de pérdida y precisión
    test_loss /= num_batches
    correct /= size
    return test_loss, correct


# Función de evaluación para obtener las métricas
def eval_model(dataloader, model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            predicted = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Convertir a arrays de NumPy para calcular las métricas
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calcular las métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

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
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 32
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Crear la red neuronal
    model = SigmoidNNWithHiddenLayer(X_train_tensor.shape[1])

    # Definir la función de pérdida (entropía cruzada binaria) y el optimizador (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

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
    
    y_train_full = train_full['cid']
    x_train_full = train_full.drop(columns=['cid'])
    x_test = test.drop(columns=['cid'])
    y_test = test['cid']

        # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train_full)
    X_val = scaler.transform(x_test)

    # Convertir a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_full.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test.values, dtype=torch.long)

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 32
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    # Crear la red neuronal
    model = SigmoidNNWithHiddenLayer(X_train_tensor.shape[1])

    # Definir la función de pérdida (entropía cruzada binaria) y el optimizador (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # Entrenamiento
    num_epochs = 100
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        # Ejecutar el ciclo de entrenamiento
        train_loss, train_acc = train_loop(dataloader_train, model, criterion, optimizer)

    # Evaluación del modelo con los datos de validación
    accuracy, precision, recall, f1 = eval_model(dataloader_val, model)

    print(f'Accuracy del modelo (validación): {accuracy:.4f}')
    print(f'Precision del modelo (validación): {precision:.4f}')
    print(f'Recall del modelo (validación): {recall:.4f}')
    print(f'F1 del modelo (validación): {f1:.4f}')