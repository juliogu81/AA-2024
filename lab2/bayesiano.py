import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, recall_score, precision_recall_curve, auc
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_score
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer


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
            class_prob = np.log2(self.class_probs[label])
            
            # Multiplicar por P(x_i|y) ajustado con m para cada característica categórica
            for i, value in enumerate(x):
                if value in self.feature_probs[label][i]:
                    class_prob += np.log2(self.feature_probs[label][i][value])
                else:
                    # Si el valor categórico no ha sido visto en el entrenamiento, aplicar suavizado
                    p_xi = 1 / len(self.feature_probs[label][i])
                    class_prob += np.log2((self.m * p_xi) / (self.class_counts[label] + self.m))
            
            class_probs[label] = class_prob
        
        # Predecir la clase con la mayor probabilidad
        predicted_label = max(class_probs, key=class_probs.get)
    
        return predicted_label


    def predict_probs(self, x):
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
        
        # Retornar la probabilidad de la clase positiva
        probabilidad_total = class_probs[1] + class_probs[0]
        return class_probs[1] / probabilidad_total




if __name__ == "__main__":
    # Cargar el dataset con encabezados
    DATASET_FILE = 'lab1_dataset.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=",")

    # Guardar los nombres de las columnas
    column_names = dataset.columns[1:]  # Excluir la primera columna (ID del paciente)

    # Eliminar la primera columna (ID del paciente)
    dataset = dataset.drop(dataset.columns[0], axis=1)

    # Definir las columnas categóricas manualmente
    columnas_categoricas = [1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]

    # Extraer los atributos y la etiqueta
    atributos = dataset.iloc[:, 1:].values  # Todas las columnas menos la primera (la etiqueta)
    etiqueta = dataset.iloc[:, 0].values    # Primera columna como etiqueta

    # Aplicar la discretización solo a las columnas continuas
    for col in range(atributos.shape[1]):
        if col not in columnas_categoricas:
            # Crear un KBinsDiscretizer para discretizar los valores continuos de esta columna en específico
            discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
            
            # Discretizar la columna continua
            atributos[:, col:col+1] = discretizer.fit_transform(atributos[:, col:col+1])

    # Convertir `etiqueta` a una forma de columna para la concatenación
    etiqueta_columna = etiqueta.reshape(-1, 1)

    # Conatenar `etiqueta` con los atributos discretizados
    dataset_discretizado = np.hstack([etiqueta_columna, atributos])

    # Ajustar los nombres de las columnas
    column_names_discretizado = ['etiqueta'] + list(column_names[1:])

    # Convertir el array discretizado en un DataFrame
    dataset_discretizado_df = pd.DataFrame(dataset_discretizado, columns=column_names_discretizado)

    #Eliminar columnas tras prueba chi-cuadrado
    dataset_discretizado_df = dataset_discretizado_df.drop(columns=['wtkg','hemo','homo', 'karnof', 'oprior', 'zprior', 'preanti', 'gender','cd40','cd820'])

    # Extraer los atributos y la etiqueta
    atributos = dataset_discretizado_df.iloc[:, 1:].values  # Todas las columnas menos la primera (la etiqueta)
    etiqueta = dataset_discretizado_df.iloc[:, 0].values    # Primera columna como etiqueta

    # Validación cruzada manual con nuestro algoritmo bayesiano implementado
    precisiones_m1_train = []
    precisiones_m1_test = []
    precisiones_m10_train = []
    precisiones_m10_test = []
    precisiones_m100_train = []
    precisiones_m100_test = []
    precisiones_m1000_train = []
    precisiones_m1000_test = []

    # Inicializar listas para almacenar los valores de recall
    recalls_m1_train = []
    recalls_m1_test = []
    recalls_m10_train = []
    recalls_m10_test = []
    recalls_m100_train = []
    recalls_m100_test = []
    recalls_m1000_train = []
    recalls_m1000_test = []


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

        # Calcular recall en datos de entrenamiento
        recall_m1_train = recall_score(etiqueta_train, predicciones_m1_train, average='macro')
        recalls_m1_train.append(recall_m1_train)


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

        # Calcular recall en datos de prueba
        recall_m1_test = recall_score(etiqueta_test, predicciones_m1_test, average='macro')
        recalls_m1_test.append(recall_m1_test)

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

        # Calcular recall en datos de entrenamiento
        recall_m10_train = recall_score(etiqueta_train, predicciones_m10_train, average='macro')
        recalls_m10_train.append(recall_m10_train)

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

        # Calcular recall en datos de prueba
        recall_m10_test = recall_score(etiqueta_test, predicciones_m10_test, average='macro')
        recalls_m10_test.append(recall_m10_test)

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

        # Calcular recall en datos de entrenamiento
        recall_m100_train = recall_score(etiqueta_train, predicciones_m100_train, average='macro')
        recalls_m100_train.append(recall_m100_train)

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

        # Calcular recall en datos de prueba
        recall_m100_test = recall_score(etiqueta_test, predicciones_m100_test, average='macro')
        recalls_m100_test.append(recall_m100_test)

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

        # Calcular recall en datos de entrenamiento
        recall_m1000_train = recall_score(etiqueta_train, predicciones_m1000_train, average='macro')
        recalls_m1000_train.append(recall_m1000_train)

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

        # Calcular recall en datos de prueba
        recall_m1000_test = recall_score(etiqueta_test, predicciones_m1000_test, average='macro')
        recalls_m1000_test.append(recall_m1000_test)

    # Calcular el promedio de recall y precisión para cada valor de m
    precision_promedio_m1_train = np.mean(precisiones_m1_train)
    precision_desviacion_m1_train = np.std(precisiones_m1_train)
    recall_promedio_m1_train = np.mean(recalls_m1_train)
    
    precision_promedio_m1_test = np.mean(precisiones_m1_test)
    precision_desviacion_m1_test = np.std(precisiones_m1_test)
    recall_promedio_m1_test = np.mean(recalls_m1_test)

    
    precision_promedio_m10_train = np.mean(precisiones_m10_train)
    precision_desviacion_m10_train = np.std(precisiones_m10_train)    
    recall_promedio_m10_train = np.mean(recalls_m10_train)


    precision_promedio_m10_test = np.mean(precisiones_m10_test)
    precision_desviacion_m10_test = np.std(precisiones_m10_test)
    recall_promedio_m10_test = np.mean(recalls_m10_test)

    
    precision_promedio_m100_train = np.mean(precisiones_m100_train)
    precision_desviacion_m100_train = np.std(precisiones_m100_train)    
    recall_promedio_m100_train = np.mean(recalls_m100_train)

    precision_promedio_m100_test = np.mean(precisiones_m100_test)
    precision_desviacion_m100_test = np.std(precisiones_m100_test)
    recall_promedio_m100_test = np.mean(recalls_m100_test)

    precision_promedio_m1000_train = np.mean(precisiones_m1000_train)
    precision_desviacion_m1000_train = np.std(precisiones_m1000_train)
    recall_promedio_m1000_train = np.mean(recalls_m1000_train)

    precision_promedio_m1000_test = np.mean(precisiones_m1000_test)
    precision_desviacion_m1000_test = np.std(precisiones_m1000_test)   
    recall_promedio_m1000_test = np.mean(recalls_m1000_test)

    # Mostrar resultados
    print(
        "\n------ Resultados de la Validación Cruzada para m = 1, 10, 100 y 1000------"
    )
    
    print(
        f"Promedio de precisión en datos de entrenamiento con m = 1: {precision_promedio_m1_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 1: {precision_desviacion_m1_train * 100:.2f}%"
    )
    print(f"Promedio de recall en datos de entrenamiento con m = 1: {recall_promedio_m1_train * 100:.2f}%")
    print(
        f"Promedio de precisión en datos de prueba con m = 1: {precision_promedio_m1_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 1: {precision_desviacion_m1_test * 100:.2f}%"
    )
    print(f"Promedio de recall en datos de prueba con m = 1: {recall_promedio_m1_test * 100:.2f}%")


    print(
        f"Promedio de precisión en datos de entrenamiento con m = 10: {precision_promedio_m10_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 10: {precision_desviacion_m10_train * 100:.2f}%"
    )
    print(f"Promedio de recall en datos de entrenamiento con m = 10: {recall_promedio_m10_train * 100:.2f}%")
    print(
        f"Promedio de precisión en datos de prueba con m = 10: {precision_promedio_m10_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 10: {precision_desviacion_m10_test * 100:.2f}%"
    )    
    print(f"Promedio de recall en datos de prueba con m = 10: {recall_promedio_m10_test * 100:.2f}%")


    print(
        f"Promedio de precisión en datos de entrenamiento con m = 100: {precision_promedio_m100_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 100: {precision_desviacion_m100_train * 100:.2f}%"
    )
    print(f"Promedio de recall en datos de entrenamiento con m = 100: {recall_promedio_m100_train * 100:.2f}%")
    print(
        f"Promedio de precisión en datos de prueba con m = 100: {precision_promedio_m100_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 100: {precision_desviacion_m100_test * 100:.2f}%"
    )
    print(f"Promedio de recall en datos de prueba con m = 100: {recall_promedio_m100_test * 100:.2f}%")


    print(
        f"Promedio de precisión en datos de entrenamiento con m = 1000: {precision_promedio_m1000_train * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de entrenamiento con m = 1000: {precision_desviacion_m1000_train * 100:.2f}%"
    )
    print(f"Promedio de recall en datos de entrenamiento con m = 1000: {recall_promedio_m1000_train * 100:.2f}%")
    print(
        f"Promedio de precisión en datos de prueba con m = 1000: {precision_promedio_m1000_test * 100:.2f}%"
    )
    print(
        f"Desviación estándar de precisión en datos de prueba con m = 1000: {precision_desviacion_m1000_test * 100:.2f}%"
    )    
    print(f"Promedio de recall en datos de prueba con m = 1000: {recall_promedio_m1000_test * 100:.2f}%")


    # Mostrar resultados de curvas de precisión-recall
    print(
        "\n------ Resultados de curva precision-recall para m = 1, 10, 100 y 1000------"
    )


    # Dividir el dataset en entrenamiento y prueba
    train, test = model_selection.train_test_split(dataset_discretizado_df, test_size=0.2, random_state=42)


    # Extraer los atributos y la etiqueta
    atributos_train = train.iloc[:, 1:].values  # Todas las columnas menos la primera (la etiqueta)
    etiqueta_train = train.iloc[:, 0].values    # Primera columna como etiqueta

    # Extraer los atributos y la etiqueta
    atributos_test = test.iloc[:, 1:].values  # Todas las columnas menos la primera (la etiqueta)
    etiqueta_test = test.iloc[:, 0].values    # Primera columna como etiqueta
   

    # Entrenar el modelo con m = 1
    bayes_m1 = Bayesiano()
    bayes_m1.fit(atributos_train, etiqueta_train, 1)

    # Entrenar el modelo con m = 10
    bayes_m10 = Bayesiano()
    bayes_m10.fit(atributos_train, etiqueta_train, 10)

    # Entrenar el modelo con m = 100
    bayes_m100 = Bayesiano()
    bayes_m100.fit(atributos_train, etiqueta_train, 100)

    # Entrenar el modelo con m = 1000
    bayes_m1000 = Bayesiano()
    bayes_m1000.fit(atributos_train, etiqueta_train, 1000)

    
    # Predecir y calcular precisión en datos de entrenamiento
    predicciones_m1 = [bayes_m1.predict_probs(x) for x in atributos_test]

    predicciones_m10 = [bayes_m10.predict_probs(x) for x in atributos_test]

    predicciones_m100 = [bayes_m100.predict_probs(x) for x in atributos_test]

    predicciones_m1000 = [bayes_m1000.predict_probs(x) for x in atributos_test]


    # Calcular la curva de precisión-recall para cada valor de m
    precisions_m1, recalls_m1, thresholds_m1 = precision_recall_curve(etiqueta_test, predicciones_m1)
    precisions_m10, recalls_m10, thresholds_m10 = precision_recall_curve(etiqueta_test, predicciones_m10)
    precisions_m100, recalls_m100, thresholds_m100 = precision_recall_curve(etiqueta_test, predicciones_m100)
    precisions_m1000, recalls_m1000, thresholds_m1000 = precision_recall_curve(etiqueta_test, predicciones_m1000)

    auc_m1 = auc(recalls_m1, precisions_m1)
    auc_m10 = auc(recalls_m10, precisions_m10)
    auc_m100 = auc(recalls_m100, precisions_m100)
    auc_m1000 = auc(recalls_m1000, precisions_m1000)


    # Graficar la curva precisión-recall para cada valor de m
    plt.plot(recalls_m1, precisions_m1, marker='.', label='m = 1')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall con m = 1. AUC = ' + str(auc_m1))
    plt.legend()
    plt.show()

    plt.plot(recalls_m10, precisions_m10, marker='.', label='m = 10')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall con m = 10. AUC = ' + str(auc_m10))
    plt.legend()
    plt.show()

    plt.plot(recalls_m100, precisions_m100, marker='.', label='m = 100')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall con m = 100. AUC = ' + str(auc_m100))
    plt.legend()
    plt.show()

    plt.plot(recalls_m1000, precisions_m1000, marker='.', label='m = 1000')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall con m = 1000. AUC = ' + str(auc_m1000))
    plt.legend()
    plt.show()

