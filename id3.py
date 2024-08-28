import numpy as np
from math import log2
import pandas as pd

def calcular_entropia(etiqueta):
    # Contar la frecuencia de cada etiqueta
    frecuencia = {}
    for item in etiqueta:
        if item in frecuencia:
            frecuencia[item] += 1
        else:
            frecuencia[item] = 1
    
    # Calcular la probabilidad de cada etiqueta
    total = len(etiqueta)
    proporciones = [frecuencia[item] / total for item in frecuencia]
    
    # Calcular la entropía
    return -sum(p * log2(p) for p in proporciones)

def calcular_ganancia(atributos, etiqueta, atributo_a_calcular):
    # atributos es el conjunto de características
    # etiqueta es la lista de etiquetas de clase
    # atributo_a_calcular es la columna de características sobre la cual calcular la ganancia
    entropia_original = calcular_entropia(etiqueta)
    valores, conteos = np.unique(atributos[:, atributo_a_calcular], return_counts=True)
    entropia_atributo = sum((conteos[i] / sum(conteos)) * calcular_entropia(etiqueta[atributos[:, atributo_a_calcular] == v]) for i, v in enumerate(valores))
    return entropia_original - entropia_atributo


def _etiqueta_mas_comun(self, etiqueta):
        # Encuentra la etiqueta que aparece más frecuentemente
        (valores, conteos) = np.unique(etiqueta, return_counts=True)
        indice = np.argmax(conteos)
        return valores[indice]


def train_test_split(dataset, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(dataset))
    test_indices = indices[:int(test_size * len(dataset))]
    train_indices = indices[int(test_size * len(dataset)):]
    train = dataset.iloc[train_indices]
    test = dataset.iloc[test_indices]
    return train, test

class ArbolDecision:
    def __init__(self):
        self.arbol = None
    
    def fit(self, atributos, etiqueta):
        self.arbol = self._construir_arbol(atributos, etiqueta, max_range_split)
    
    def _construir_arbol(self, atributos, etiqueta):
        # Si todas las etiquetas son iguales, devuelve la etiqueta
        if len(np.unique(etiqueta)) == 1:
            return etiqueta[0]
        
        # Si no quedan más atributos, devolver la etiqueta más común
        if atributos.shape[1] == 0:
            return self._etiqueta_mas_comun(etiqueta)


        # Encuentra la característica con la mayor ganancia de información
        ganancias = [calcular_ganancia(atributos, etiqueta, i) for i in range(atributos.shape[1])]
        mejor_atributo = np.argmax(ganancias)
        
        # Crea los nodos del árbol
        arbol = {mejor_atributo: {}}
        values = np.unique(atributos[:, mejor_atributo])
        for v in values:
            subset_X = atributos[atributos[:, mejor_atributo] == v]
            subset_y = etiqueta[atributos[:, mejor_atributo] == v]

            # Eliminar el mejor atributo del subconjunto antes de la recursión
            subset_X = np.delete(subset_X, mejor_atributo, axis=1)

            subtree = self._construir_arbol(subset_X, subset_y)
            arbol[mejor_atributo][v] = subtree
        
        return arbol
    

    def _es_binario(self, atributo):
        # Verifica si el atributo es binario
        return len(np.unique(atributo)) == 2
    
    def _encontrar_puntos_corte(self, atributo, etiqueta):
        # Encuentra puntos de corte óptimos
        puntos_corte = []
        # Ordena los atributos y etiquetas juntos
        datos_ordenados = sorted(zip(atributo, etiqueta), key=lambda x: x[0])
        for i in range(1, len(datos_ordenados)):
            if datos_ordenados[i-1][1] != datos_ordenados[i][1]:
                punto_corte = (datos_ordenados[i-1][0] + datos_ordenados[i][0]) / 2
                puntos_corte.append(punto_corte)
        return puntos_corte

    def predict(self, X):
        # Maneja tanto una lista de ejemplos como un solo ejemplo
        if X.ndim == 1:
            return self._predecir_ejemplo(X, self.arbol)
        else:
            return [self._predecir_ejemplo(ejemplo, self.arbol) for ejemplo in X]

    def _predecir_ejemplo(self, ejemplo, arbol):
        if not isinstance(arbol, dict):
            return arbol

        atributo = list(arbol.keys())[0]
        valor = ejemplo[atributo]

        if valor in arbol[atributo]:
            subarbol = arbol[atributo][valor]
            return self._predecir_ejemplo(ejemplo, subarbol)
        else:
            return None






tree = ArbolDecision()

DATASET_FILE = 'qsar_oral_toxicity.csv'

dataset = pd.read_csv(DATASET_FILE, sep=";", header=None).add_prefix("c")
print(
    f"{dataset.shape[0]} records read from {DATASET_FILE}\n{dataset.shape[1]} attributes found"
)
dataset.head(10)

dataset.c24.value_counts()



train, test = train_test_split(dataset, test_size=0.85, random_state=42)
print(f"{train.shape[0]} samples for training, {test.shape[0]} samples for testing")
train.head(10)

atributos = train.iloc[:, :-1].values
etiqueta = train.iloc[:, -1].values

tree.fit(atributos, etiqueta)

#validar el arbol con test
atributos_test = test.iloc[:, :-1].values
etiqueta_test = test.iloc[:, -1].values

predicciones = [tree.predict(x) for x in atributos_test]
precision = sum(predicciones == etiqueta_test) / len(etiqueta_test)
print(precision)
