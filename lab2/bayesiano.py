import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_score
from collections import defaultdict


class Bayesiano:
    def __init__(self):
        self.class_probs = {}  # Probabilidades P(y)
        self.feature_probs = defaultdict(lambda: defaultdict(dict))  # Probabilidades P(x_i|y)
        self.class_counts = defaultdict(int)  # Conteos de clase para P(y)
        
    
    def fit(self, X, y, m=1):
        n_samples, n_features = X.shape
        self.class_labels = np.unique(y)
        self.m = m  # Tamaño equivalente de muestra
        
        # Calcular conteos de clase P(y)
        for label in self.class_labels:
            self.class_counts[label] = np.sum(y == label)
            self.class_probs[label] = self.class_counts[label] / n_samples
            
            # Calcular conteos de características P(x_i|y)
            X_label = X[y == label]
            for i in range(n_features):
                feature_counts = defaultdict(int)
                # Contar la frecuencia de cada valor de la característica para la clase 'label'
                for value in X_label[:, i]:
                    feature_counts[value] += 1
                
                total_counts = len(X_label[:, i])
                total_values = len(np.unique(X[:, i]))
                
                # Almacenar probabilidades ajustadas con el hiperparámetro m
                for value, count in feature_counts.items():
                    # P(x_i) para el valor de la característica en todos los datos (no condicionado por clase)
                    p_xi = np.sum(X[:, i] == value) / n_samples
                    # Suavizado con el hiperparámetro m
                    self.feature_probs[label][i][value] = (count + self.m * p_xi) / (total_counts + self.m)
    
    def predict(self, x):
        class_probs = {}
        
        # Calcular P(y|X) para cada clase
        for label in self.class_labels:
            # Comenzar con P(y)
            class_prob = self.class_probs[label]
            
            # Multiplicar por P(x_i|y) ajustado con m para cada característica categórica
            for i, value in enumerate(x):
                if value in self.feature_probs[label][i]:
                    class_prob *= self.feature_probs[label][i][value]
                else:
                    # Si el valor categórico no ha sido visto en el entrenamiento, aplicar suavizado
                    p_xi = 1 / len(self.feature_probs[label][i])
                    class_prob *= (self.m * p_xi) / (self.class_counts[label] + self.m)
            
            class_probs[label] = class_prob
        
        # Predecir la clase con la mayor probabilidad
        predicted_label = max(class_probs, key=class_probs.get)
    
        return predicted_label





if __name__ == "__main__":
    # Leer el archivo
    DATASET_FILE = "lab1_dataset.csv"
    dataset = pd.read_csv(DATASET_FILE, sep=",").add_prefix("c")

    # Se elimina del dataset la primera columna ya que no es un atributo. Corresponde al ID del paciente
    dataset = dataset.drop(dataset.columns[0], axis=1)

    # Definir atributos y etiqueta desde el dataset completo
    atributos = dataset.iloc[:, 1:].values  # Todas las columnas menos la primera (función objetivo)
    etiqueta = dataset.iloc[:, 0].values  # Primera columna como etiqueta

    # Validación cruzada manual con nuestro algoritmo bayesiano implementado
    precisiones_m1_train = []
    precisiones_m1_test = []
    precisiones_m10_train = []
    precisiones_m10_test = []
    precisiones_m100_train = []
    precisiones_m100_test = []
    precisiones_m1000_train = []
    precisiones_m1000_test = []

    # Preparar validación cruzada de 5 pliegues
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n------ Resultados de pliegues en algoritmo bayesiano con m = 1, 10, 100 y 1000------")

    for fold, (train_index, test_index) in enumerate(kf.split(atributos), 1):
        atributos_train, atributos_test = atributos[train_index], atributos[test_index]
        etiqueta_train, etiqueta_test = etiqueta[train_index], etiqueta[test_index]

        print(f"\n------ Pliegue {fold} ------")

        # Entrenar el modelo con m = 1
        bayes_m1 = Bayesiano()
        bayes_m1.fit(atributos_train, etiqueta_train, 1)

        # Predecir y calcular precisión en datos de entrenamiento
        predicciones_m1_train = [bayes_m1.predict(x) for x in atributos_train]
        precision_m1_train = np.sum(
            np.array(predicciones_m1_train) == etiqueta_train
        ) / len(etiqueta_train)
        precisiones_m1_train.append(precision_m1_train)
        print(
            f"Precisión en datos de entrenamiento con m = 1: {precision_m1_train * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m1_train = confusion_matrix(etiqueta_train, predicciones_m1_train)
        print(f"\nMatriz de confusión en datos de entrenamiento con m = 1:\n{matriz_confusion_m1_train}\n")

        # Predecir y calcular precisión en datos de prueba
        predicciones_m1_test = [bayes_m1.predict(x) for x in atributos_test]
        precision_m1_test = np.sum(
            np.array(predicciones_m1_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m1_test.append(precision_m1_test)
        print(
            f"Precisión en datos de prueba con m = 1: {precision_m1_test * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m1_test = confusion_matrix(etiqueta_test, predicciones_m1_test)
        print(f"\nMatriz de confusión en datos de prueba con m = 1:\n{matriz_confusion_m1_test}\n")

        # Entrenar el modelo con m = 10
        bayes_m10 = Bayesiano()
        bayes_m10.fit(atributos_train, etiqueta_train, 10)

        # Predecir y calcular precisión en datos de entrenamiento
        predicciones_m10_train = [bayes_m10.predict(x) for x in atributos_train]
        precision_m10_train = np.sum(
            np.array(predicciones_m10_train) == etiqueta_train
        ) / len(etiqueta_train)
        precisiones_m10_train.append(precision_m10_train)
        print(
            f"Precisión en datos de entrenamiento con m = 10: {precision_m10_train * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m10_train = confusion_matrix(etiqueta_train, predicciones_m10_train)
        print(f"\nMatriz de confusión en datos de entrenamiento con m = 10:\n{matriz_confusion_m10_train}\n")

        # Predecir y calcular precisión en datos de prueba
        predicciones_m10_test = [bayes_m10.predict(x) for x in atributos_test]
        precision_m10_test = np.sum(
            np.array(predicciones_m10_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m10_test.append(precision_m10_test)
        print(
            f"Precisión en datos de prueba con m = 10: {precision_m10_test * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m10_test = confusion_matrix(etiqueta_test, predicciones_m10_test)
        print(f"\nMatriz de confusión en datos de prueba con m = 10:\n{matriz_confusion_m10_test}\n")

        # Entrenar el modelo con m = 100
        bayes_m100 = Bayesiano()
        bayes_m100.fit(atributos_train, etiqueta_train, 100)

        # Predecir y calcular precisión en datos de entrenamiento
        predicciones_m100_train = [bayes_m100.predict(x) for x in atributos_train]
        precision_m100_train = np.sum(
            np.array(predicciones_m100_train) == etiqueta_train
        ) / len(etiqueta_train)
        precisiones_m100_train.append(precision_m100_train)
        print(
            f"Precisión en datos de entrenamiento con m = 100: {precision_m100_train * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m100_train = confusion_matrix(etiqueta_train, predicciones_m100_train)
        print(f"\nMatriz de confusión en datos de entrenamiento con m = 100:\n{matriz_confusion_m100_train}\n")

        # Predecir y calcular precisión en datos de prueba
        predicciones_m100_test = [bayes_m100.predict(x) for x in atributos_test]
        precision_m100_test = np.sum(
            np.array(predicciones_m100_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m100_test.append(precision_m100_test)
        print(
            f"Precisión en datos de prueba con m = 100: {precision_m100_test * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m100_test = confusion_matrix(etiqueta_test, predicciones_m100_test)
        print(f"\nMatriz de confusión en datos de prueba con m = 100:\n{matriz_confusion_m100_test}\n")

        # Entrenar el modelo con m = 1000
        bayes_m1000 = Bayesiano()
        bayes_m1000.fit(atributos_train, etiqueta_train, 1000)

        # Predecir y calcular precisión en datos de entrenamiento
        predicciones_m1000_train = [bayes_m1000.predict(x) for x in atributos_train]
        precision_m1000_train = np.sum(
            np.array(predicciones_m1000_train) == etiqueta_train
        ) / len(etiqueta_train)
        precisiones_m1000_train.append(precision_m1000_train)
        print(
            f"Precisión en datos de entrenamiento con m = 1000: {precision_m1000_train * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m1000_train = confusion_matrix(etiqueta_train, predicciones_m1000_train)
        print(f"\nMatriz de confusión en datos de entrenamiento con m = 1000:\n{matriz_confusion_m1000_train}\n")

        # Predecir y calcular precisión en datos de prueba
        predicciones_m1000_test = [bayes_m1000.predict(x) for x in atributos_test]
        precision_m1000_test = np.sum(
            np.array(predicciones_m1000_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m1000_test.append(precision_m1000_test)
        print(
            f"Precisión en datos de prueba con m = 1000: {precision_m1000_test * 100:.2f}%"
        )

        # Generar la matriz de confusión
        matriz_confusion_m1000_test = confusion_matrix(etiqueta_test, predicciones_m1000_test)
        print(f"\nMatriz de confusión en datos de prueba con m = 1000:\n{matriz_confusion_m1000_test}\n")


    # Calcular la precisión promedio de los 5 pliegues para cada modelo
    precision_promedio_m1_train = np.mean(precisiones_m1_train)
    precision_desviacion_m1_train = np.std(precisiones_m1_train)

    precision_promedio_m1_test = np.mean(precisiones_m1_test)
    precision_desviacion_m1_test = np.std(precisiones_m1_test)

    precision_promedio_m10_train = np.mean(precisiones_m10_train)
    precision_desviacion_m10_train = np.std(precisiones_m10_train)

    precision_promedio_m10_test = np.mean(precisiones_m10_test)
    precision_desviacion_m10_test = np.std(precisiones_m10_test)

    precision_promedio_m100_train = np.mean(precisiones_m100_train)
    precision_desviacion_m100_train = np.std(precisiones_m100_train)

    precision_promedio_m100_test = np.mean(precisiones_m100_test)
    precision_desviacion_m100_test = np.std(precisiones_m100_test)

    precision_promedio_m1000_train = np.mean(precisiones_m1000_train)
    precision_desviacion_m1000_train = np.std(precisiones_m1000_train)

    precision_promedio_m1000_test = np.mean(precisiones_m1000_test)
    precision_desviacion_m1000_test = np.std(precisiones_m1000_test)


    print(
        "\n------ Resultados de la Validación Cruzada para m = 1, 10, 100 y 1000------"
    )
    print(
        f"Promedio de precisión en datos de entrenamiento con m = 1: {precision_promedio_m1_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 1: {precision_desviacion_m1_train * 100:.2f}%"
    )
    print(
        f"Promedio de precisión en datos de prueba con m = 1: {precision_promedio_m1_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 1: {precision_desviacion_m1_test * 100:.2f}%"
    )

    print(
        f"Promedio de precisión en datos de entrenamiento con m = 10: {precision_promedio_m10_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 10: {precision_desviacion_m10_train * 100:.2f}%"
    )
    print(
        f"Promedio de precisión en datos de prueba con m = 10: {precision_promedio_m10_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 10: {precision_desviacion_m10_test * 100:.2f}%"
    )

    print(
        f"Promedio de precisión en datos de entrenamiento con m = 100: {precision_promedio_m100_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 100: {precision_desviacion_m100_train * 100:.2f}%"
    )
    print(
        f"Promedio de precisión en datos de prueba con m = 100: {precision_promedio_m100_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 100: {precision_desviacion_m100_test * 100:.2f}%"
    )

    print(
        f"Promedio de precisión en datos de entrenamiento con m = 1000: {precision_promedio_m1000_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 1000: {precision_desviacion_m1000_train * 100:.2f}%"
    )
    print(
        f"Promedio de precisión en datos de prueba con m = 1000: {precision_promedio_m1000_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 1000: {precision_desviacion_m1000_test * 100:.2f}%"
    )
