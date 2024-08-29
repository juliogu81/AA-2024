import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Leer el archivo y preparar los datos
DATASET_FILE = 'lab1_dataset.csv'
dataset = pd.read_csv(DATASET_FILE, sep=",", header=None).add_prefix("c")
dataset = dataset.drop(dataset.columns[0], axis=1)

# Convertir todos los atributos a valores numéricos
le = preprocessing.LabelEncoder()
for column in dataset.columns:
    dataset[column] = le.fit_transform(dataset[column])

# Dividir en conjunto de entrenamiento y prueba
train, test = model_selection.train_test_split(dataset, test_size=0.2, random_state=42)

# Preparar los datos para el entrenamiento
atributos = train.iloc[:, 1:].values
etiqueta = train.iloc[:, 0].values

# Preparar los datos para la prueba
atributos_test = test.iloc[:, 1:].values
etiqueta_test = test.iloc[:, 0].values

# Entrenar y evaluar DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(atributos, etiqueta)

# Hacer predicciones y calcular la precisión
predicciones_dt = dt_classifier.predict(atributos_test)
precision_dt = np.sum(predicciones_dt == etiqueta_test) / len(etiqueta_test)
print(f"Precisión del DecisionTreeClassifier: {precision_dt}")

# Entrenar y evaluar RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(atributos, etiqueta)

# Hacer predicciones y calcular la precisión
predicciones_rf = rf_classifier.predict(atributos_test)
precision_rf = np.sum(predicciones_rf == etiqueta_test) / len(etiqueta_test)
print(f"Precisión del RandomForestClassifier: {precision_rf}")
