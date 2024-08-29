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
        self.mask = None
    
    def fit(self, atributos, etiqueta, max_range_split):
        self.mask = mask = np.ones(atributos.shape[1], dtype=bool)
        self.arbol = self._construir_arbol(atributos, etiqueta, max_range_split)
    
    def _construir_arbol(self, atributos, etiqueta, max_range_split):
        # Si todas las etiquetas son iguales, devuelve la etiqueta
        if len(np.unique(etiqueta)) == 1:
            return etiqueta[0]
        
        # Si no quedan más atributos, devolver la etiqueta más común
        if self.mask.sum() == 0:
            return self._etiqueta_mas_comun(etiqueta)

        es_continuo = False
        # Encuentra la característica con la mayor ganancia de información. Si es continua se debe dividir en max_range_split
        ganancia_max = - float('inf')
        mejor_atributo = None
        for atri in range(atributos.shape[1]):
            if self.mask[atri]:   
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
                
                # Quitamos el mejor atributo de la mascara antes de la recursión
                self.mask[mejor_atributo] = False
                
                subtree = self._construir_arbol(subset_X, subset_y, max_range_split)
                arbol[mejor_atributo][v] = subtree
        
        else:
            # Para atributos continuos, almacenamos los puntos de corte y los intervalos
            puntos_corte_optimos = sorted(puntos_corte_optimos)
            for i in range(len(puntos_corte_optimos) + 1):
                if i == 0:
                    subset_X = atributos[atributos[:, mejor_atributo] <= puntos_corte_optimos[i]]
                    subset_y = etiqueta[atributos[:, mejor_atributo] <= puntos_corte_optimos[i]]
                    intervalo = (-float('inf'), puntos_corte_optimos[i])
                elif i == len(puntos_corte_optimos):
                    subset_X = atributos[atributos[:, mejor_atributo] > puntos_corte_optimos[i - 1]]
                    subset_y = etiqueta[atributos[:, mejor_atributo] > puntos_corte_optimos[i - 1]]
                    intervalo = (puntos_corte_optimos[i - 1], float('inf'))
                else:
                    subset_X = atributos[(atributos[:, mejor_atributo] > puntos_corte_optimos[i - 1]) & (atributos[:, mejor_atributo] <= puntos_corte_optimos[i])]
                    subset_y = etiqueta[(atributos[:, mejor_atributo] > puntos_corte_optimos[i - 1]) & (atributos[:, mejor_atributo] <= puntos_corte_optimos[i])]
                    intervalo = (puntos_corte_optimos[i - 1], puntos_corte_optimos[i])
                
                self.mask[mejor_atributo] = False
                subtree = self._construir_arbol(subset_X, subset_y, max_range_split)
                arbol[mejor_atributo][intervalo] = subtree
        
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

    def _etiqueta_mas_comun(self, etiqueta):
        # Encuentra la etiqueta que aparece más frecuentemente
        (valores, conteos) = np.unique(etiqueta, return_counts=True)
        indice = np.argmax(conteos)
        return valores[indice]

    def predict(self, X):
        if self.arbol is None:
            raise ValueError("El árbol no ha sido entrenado. Llama a `fit` primero.")
        
        # Si X es un solo dato (un vector), tenemos que convertirlo en una matriz de una sola fila
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predicciones = np.array([self._predecir_un_dato(x) for x in X])
        return predicciones

    def _predecir_un_dato(self, x):
        arbol = self.arbol
        while isinstance(arbol, dict):
            atributo = list(arbol.keys())[0]
            valor = x[atributo]
            
            if atributo in arbol:
                if isinstance(arbol[atributo], dict):
                    if isinstance(list(arbol[atributo].keys())[0], tuple):
                        intervalo = self._encontrar_intervalo(valor, atributo)
                        arbol = arbol[atributo].get(intervalo, None)
                    else:
                        arbol = arbol[atributo].get(valor, None)
                        if arbol is None:
                            return None
                else:
                    arbol = arbol[atributo].get(valor, None)
                    if arbol is None:
                        return None
            else:
                return arbol
            
        return arbol

    def _encontrar_intervalo(self, valor, atributo):
        # Determina el intervalo al que pertenece el valor
        intervalos = sorted(self.arbol[atributo].keys())
        for intervalo in intervalos:
            if intervalo[0] < valor <= intervalo[1]:
                return intervalo
        return intervalos[-1]  # Valor fuera del último intervalo





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
