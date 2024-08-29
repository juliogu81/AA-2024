import numpy as np
from math import log2
import pandas as pd
from itertools import combinations

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

def calcular_ganancia_discretizado(intervalos, etiqueta):
    # Se considera que 'atributo_a_calcular' es 'None' para valores discretos
    return calcular_ganancia(intervalos, etiqueta, atributo_a_calcular=None)

def calcular_ganancia(atributos, etiqueta, atributo_a_calcular):
    entropia_original = calcular_entropia(etiqueta)
    
    if atributo_a_calcular is None:
        # En el caso de atributos discretos, usamos los intervalos calculados
        valores, conteos = np.unique(atributos, return_counts=True)
        entropia_atributo = sum((conteos[i] / sum(conteos)) * calcular_entropia(etiqueta[atributos == v]) for i, v in enumerate(valores))
    else:
        # En el caso de atributos continuos, usamos la función calculada anteriormente
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
    
    def fit(self, atributos, etiqueta, max_range_split):
        self.arbol = self._construir_arbol(atributos, etiqueta, max_range_split)
    
    def _construir_arbol(self, atributos, etiqueta, max_range_split):
        # Si todas las etiquetas son iguales, devuelve la etiqueta
        if len(np.unique(etiqueta)) == 1:
            return etiqueta[0]
        
        # Si no quedan más atributos, devolver la etiqueta más común
        if atributos.shape[1] == 0:
            return self._etiqueta_mas_comun(etiqueta)

        es_continuo = False
        # Encuentra la característica con la mayor ganancia de información. Si es continua se debe dividir en max_range_split
        ganancia_max = - float('inf')
        mejor_atributo = None
        print(atributos.shape[1])
        for atri in range(atributos.shape[1]):
            if self._es_binario(atributos[:, atri]):
                ganancia = calcular_ganancia(atributos, etiqueta, atri)
                if ganancia > ganancia_max:
                    ganancia_max = ganancia
                    mejor_atributo = atri
                    es_continuo = False
            else:
                puntos_corte = self._encontrar_mejores_puntos_corte(atributos[:, atri], etiqueta, max_range_split)
                intervalos = np.digitize(atributos[:, atri], puntos_corte)
                ganancia = calcular_ganancia_discretizado(intervalos, etiqueta)
                
                if ganancia > ganancia_max:
                    ganancia_max = ganancia
                    mejor_atributo = atri
                    es_continuo = True
                    puntos_corte_optimos = puntos_corte

        # Crea los nodos del árbol
        arbol = {mejor_atributo: {}}
        if (not es_continuo):
            values = np.unique(atributos[:, mejor_atributo])
            for v in values:
                subset_X = atributos[atributos[:, mejor_atributo] == v]
                subset_y = etiqueta[atributos[:, mejor_atributo] == v]
                
                # Eliminar el mejor atributo del subconjunto antes de la recursión
                subset_X = np.delete(subset_X, mejor_atributo, axis=1)
                
                subtree = self._construir_arbol(subset_X, subset_y, max_range_split)
                arbol[mejor_atributo][v] = subtree
        
        else:
            intervalos = np.digitize(atributos[:, mejor_atributo], puntos_corte_optimos)
            for i in range(len(puntos_corte_optimos) + 1):
                subset_X = atributos[intervalos == i]
                subset_y = etiqueta[intervalos == i]
                subset_X = np.delete(subset_X, mejor_atributo, axis=1)
                subtree = self._construir_arbol(subset_X, subset_y, max_range_split)
                arbol[mejor_atributo][i] = subtree
            
        
        return arbol

    def _es_binario(self, atributo):
        # Verifica si el atributo es binario
        return len(np.unique(atributo)) <= 2
    
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


    def _encontrar_mejores_puntos_corte(self, atributo, etiqueta, max_range_split):
        puntos_corte = self._encontrar_puntos_corte(atributo, etiqueta)
        mejor_ganancia = -float('inf')
        mejores_puntos = None
        # Probar combinaciones de puntos de corte
        for num_puntos in range(1, max_range_split):
            for combinacion in combinations(puntos_corte, num_puntos):
                combinacion = sorted(combinacion)
                intervalos = np.digitize(atributo, combinacion)
                ganancia = calcular_ganancia_discretizado(intervalos, etiqueta)
                
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejores_puntos = combinacion

        return mejores_puntos    

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

DATASET_FILE = 'lab1_dataset.csv'

dataset = pd.read_csv(DATASET_FILE, sep=",", header=0)
print(
    f"{dataset.shape[0]} records read from {DATASET_FILE}\n{dataset.shape[1]} attributes found"
)
dataset.head(10)




train, test = train_test_split(dataset, test_size=0.85, random_state=42)
print(f"{train.shape[0]} samples for training, {test.shape[0]} samples for testing")
train.head(10)

#descartamos el primer atributo
train.drop(columns=train.columns[0], inplace=True)
test.drop(columns=test.columns[0], inplace=True)


#el atributo objetivo es el primero
atributos = train.iloc[:, 1:].values
etiqueta = train.iloc[:, 0].values



tree.fit(atributos, etiqueta, 2)

#validar el arbol con test
atributos_test = test.iloc[:, 1:].values
etiqueta_test = test.iloc[:, 0].values

predicciones = [tree.predict(x) for x in atributos_test]
precision = sum(predicciones == etiqueta_test) / len(etiqueta_test)
print(precision)
