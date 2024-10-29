import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

torch.manual_seed(23)

class NeuronalNetworkV2(nn.Module):
    def __init__(self, features):
        super(NeuronalNetworkV2, self).__init__()
        # Capa de entrada
        self.input_layer = nn.Linear(features, 8)
        self.hidden_activation = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(8)
        
        # Capa oculta (una en lugar de dos)
        self.hidden_layer = nn.Linear(8, 4)
        self.batch_norm2 = nn.BatchNorm1d(4)
        
        # Capa de salida
        self.output_layer = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=0.325)  # 50% de dropout

    def forward(self, x):
        x = self.input_layer(x)
        x = self.batch_norm1(x)
        x = self.hidden_activation(x)
        
        x = self.hidden_layer(x)
        x = self.batch_norm2(x)
        x = self.hidden_activation(x)
        
        x = self.dropout(x)

        x = self.output_layer(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = torch.sigmoid(pred)
        pred_class = (pred > 0.5).type(torch.float)
        correct += (pred_class == y).type(torch.float).sum().item()

    train_loss /= len(dataloader)
    correct /= size
    return train_loss, correct


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).squeeze()
            test_loss += loss_fn(pred, y).item()

            pred = torch.sigmoid(pred)
            pred_class = (pred > 0.5).type(torch.float)
            correct += (pred_class == y).type(torch.float).sum().item()

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
            pred = torch.sigmoid(outputs)
            predicted = (pred > 0.5).type(torch.float)
            all_preds.extend(predicted.numpy())
            all_labels.extend(y_batch.numpy())

    # Convertir a arrays de NumPy para calcular las métricas
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Calcular las métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    cm = confusion_matrix (all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.title("Matriz de confusión Modelo 5.B")
    plt.show()

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    DATASET_FILE = 'lab4_dataset.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=",")
    dataset = dataset.drop(columns=['time', 'pidnum'])

    train_full, test = train_test_split(dataset, test_size=0.2, random_state=42)
    train, validation = train_test_split(train_full, test_size=0.1, random_state=42)

    X_train = train.drop(columns=['cid'])
    y_train = train['cid']
    X_val = validation.drop(columns=['cid'])
    y_val = validation['cid']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 32
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = NeuronalNetworkV2(X_train_tensor.shape[1])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Agregar el scheduler para reducir la tasa de aprendizaje si la pérdida de validación no mejora

    num_epochs = 100
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_loop(dataloader_train, model, criterion, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = test_loop(dataloader_val, model, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)


        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Entrenamiento')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validación')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Entrenamiento')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validación')
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
    y_train_tensor = torch.tensor(y_train_full.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 32
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    model = NeuronalNetworkV2(X_train_tensor.shape[1])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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