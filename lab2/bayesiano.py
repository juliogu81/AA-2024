import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_score


if __name__ == "__main__":
    # Leer el archivo
    DATASET_FILE = "lab1_dataset.csv"
    dataset = pd.read_csv(DATASET_FILE, sep=",").add_prefix("c")

    # Se elimina del dataset la primera columna ya que no es un atributo. Corresponde al ID del paciente
    dataset = dataset.drop(dataset.columns[0], axis=1)

    # Índices de columnas categóricas
    columnas_categoricas = [1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]

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

        # Predecir y calcular precisión en datos de prueba
        predicciones_m1_test = [bayes_m1.predict(x) for x in atributos_test]
        precision_m1_test = np.sum(
            np.array(predicciones_m1_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m1_test.append(precision_m1_test)
        print(
            f"Precisión en datos de prueba con m = 1: {precision_m1_test * 100:.2f}%"
        )

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

        # Predecir y calcular precisión en datos de prueba
        predicciones_m10_test = [bayes_m10.predict(x) for x in atributos_test]
        precision_m10_test = np.sum(
            np.array(predicciones_m10_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m10_test.append(precision_m10_test)
        print(
            f"Precisión en datos de prueba con m = 10: {precision_m10_test * 100:.2f}%"
        )

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

        # Predecir y calcular precisión en datos de prueba
        predicciones_m100_test = [bayes_m100.predict(x) for x in atributos_test]
        precision_m100_test = np.sum(
            np.array(predicciones_m100_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m100_test.append(precision_m100_test)
        print(
            f"Precisión en datos de prueba con m = 100: {precision_m100_test * 100:.2f}%"
        )

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

        # Predecir y calcular precisión en datos de prueba
        predicciones_m1000_test = [bayes_m1000.predict(x) for x in atributos_test]
        precision_m1000_test = np.sum(
            np.array(predicciones_m1000_test) == etiqueta_test
        ) / len(etiqueta_test)
        precisiones_m1000_test.append(precision_m1000_test)
        print(
            f"Precisión en datos de prueba con m = 1000: {precision_m1000_test * 100:.2f}%"
        )


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
