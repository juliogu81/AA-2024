import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold

def calcular_entropia(etiqueta):
    valores, conteos = np.unique(etiqueta, return_counts=True)
    probabilidades = conteos / len(etiqueta)
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    return entropia

def calcular_ganancia(atributos, etiqueta, atributo_a_calcular, puntos_corte):
    entropia_inicial = calcular_entropia(etiqueta)
    
    if puntos_corte is None or len(puntos_corte) == 0:
        valores, conteos = np.unique(atributos.iloc[:, atributo_a_calcular], return_counts=True)
        entropia_condicional = sum((conteos[i] / sum(conteos)) * calcular_entropia(etiqueta[atributos.iloc[:, atributo_a_calcular] == v]) for i, v in enumerate(valores))
    
    # Si hay un solo punto de corte
    elif len(puntos_corte) == 1:
        punto_corte = puntos_corte[0]
        menores = atributos.iloc[:, atributo_a_calcular].astype(float) <= punto_corte
        mayores = atributos.iloc[:, atributo_a_calcular].astype(float) > punto_corte
        
        if not np.any(menores) or not np.any(mayores):
            return 0  # No hay ganancia de información si un subconjunto está vacío
        
        entropia_menores = calcular_entropia(etiqueta[menores])
        entropia_mayores = calcular_entropia(etiqueta[mayores])
        
        peso_menores = np.sum(menores) / len(atributos)
        peso_mayores = np.sum(mayores) / len(atributos)
        
        entropia_condicional = (peso_menores * entropia_menores + peso_mayores * entropia_mayores)
    
    # Si hay dos puntos de corte
    elif len(puntos_corte) == 2:
        punto_corte_1 = puntos_corte[0]
        punto_corte_2 = puntos_corte[1]
        rango_1 = atributos.iloc[:, atributo_a_calcular].astype(float) <= punto_corte_1
        rango_2 = (atributos.iloc[:, atributo_a_calcular].astype(float) > punto_corte_1) & (atributos.iloc[:, atributo_a_calcular].astype(float) <= punto_corte_2)
        rango_3 = atributos.iloc[:, atributo_a_calcular].astype(float) > punto_corte_2
        
        if not np.any(rango_1) or not np.any(rango_2) or not np.any(rango_3):
            return 0  # No hay ganancia de información si un subconjunto está vacío
        
        entropia_rango_1 = calcular_entropia(etiqueta[rango_1])
        entropia_rango_2 = calcular_entropia(etiqueta[rango_2])
        entropia_rango_3 = calcular_entropia(etiqueta[rango_3])
        
        peso_rango_1 = np.sum(rango_1) / len(atributos)
        peso_rango_2 = np.sum(rango_2) / len(atributos)
        peso_rango_3 = np.sum(rango_3) / len(atributos)
        
        entropia_condicional = (peso_rango_1 * entropia_rango_1 +
                                peso_rango_2 * entropia_rango_2 +
                                peso_rango_3 * entropia_rango_3)
    
    else:
        return 0  # Caso inválido, solo permitimos 1 o 2 puntos de corte
    
    ganancia = entropia_inicial - entropia_condicional
    return ganancia

def _es_categorico(atributo, categoricos):
    return atributo in categoricos

def _encontrar_mejores_puntos_corte(atributos, atributo, indice, etiqueta, max_range_split):
    puntos_corte = []
    datos_ordenados = sorted(zip(atributo, etiqueta), key=lambda x: x[0])
    
    for i in range(1, len(datos_ordenados)):
        valor_actual, etiqueta_actual = datos_ordenados[i]
        valor_anterior, etiqueta_anterior = datos_ordenados[i - 1]
        
        if etiqueta_actual != etiqueta_anterior:
            try:
                punto_corte = (float(valor_actual) + float(valor_anterior)) / 2
                puntos_corte.append(punto_corte)
            except ValueError:
                continue
    
    if len(puntos_corte) > max_range_split:
        mejor_ganancia = -float('inf')
        # Probar combinaciones de puntos de corte
        for num_puntos in range(1, max_range_split + 1):
            for combinacion in combinations(puntos_corte, num_puntos):
                combinacion = sorted(combinacion)
                ganancia = calcular_ganancia(atributos, etiqueta, indice, combinacion)
                
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    puntos_corte = combinacion   
    
    return puntos_corte

class ArbolDecision:
    def __init__(self):
        self.arbol = None
    
    def fit(self, atributos, etiqueta, max_range_split, categoricos):
        self.arbol = self._construir_arbol(atributos, etiqueta, max_range_split, categoricos)
    
    def _construir_arbol(self, atributos, etiqueta, max_range_split, categoricos):
        if len(np.unique(etiqueta)) == 1:
            return np.unique(etiqueta)[0]
        
        if atributos.shape[1] == 0 or len(atributos) == 0:
            return np.bincount(etiqueta).argmax()
        
        mejor_atributo = None
        mejor_ganancia = -1
        mejor_puntos_corte = None
        
        for i in range(atributos.shape[1]):
            if _es_categorico(i, categoricos):
                ganancia = calcular_ganancia(atributos, etiqueta, i, None)
                puntos_corte = None
            else:
                puntos_corte = _encontrar_mejores_puntos_corte(atributos, atributos.iloc[:, i], i, etiqueta, max_range_split)
                ganancia = calcular_ganancia(atributos, etiqueta, i, puntos_corte)
            
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = i
                mejor_puntos_corte = puntos_corte
        
        if mejor_ganancia == 0 or mejor_atributo is None:
            return np.bincount(etiqueta).argmax()
        
        arbol = {}
        arbol['atributo'] = mejor_atributo
        arbol['puntos_corte'] = mejor_puntos_corte
        
        if mejor_puntos_corte is None:
            valores = np.unique(atributos.iloc[:, mejor_atributo])
            for valor in valores:
                indices = atributos.iloc[:, mejor_atributo] == valor
                if np.any(indices):
                    arbol[valor] = self._construir_arbol(atributos[indices], etiqueta[indices], max_range_split, categoricos)
        else:
            if len(mejor_puntos_corte) == 1:
                punto_corte = mejor_puntos_corte[0]
                menores = atributos.iloc[:, mejor_atributo].astype(float) <= punto_corte
                mayores = atributos.iloc[:, mejor_atributo].astype(float) > punto_corte
                if np.any(menores):
                    arbol[f'<= {punto_corte}'] = self._construir_arbol(atributos[menores], etiqueta[menores], max_range_split, categoricos)
                if np.any(mayores):
                    arbol[f'> {punto_corte}'] = self._construir_arbol(atributos[mayores], etiqueta[mayores], max_range_split, categoricos)
            elif len(mejor_puntos_corte) == 2:
                punto_corte_1 = mejor_puntos_corte[0]
                punto_corte_2 = mejor_puntos_corte[1]
                rango_1 = atributos.iloc[:, mejor_atributo].astype(float) <= punto_corte_1
                rango_2 = (atributos.iloc[:, mejor_atributo].astype(float) > punto_corte_1) & (atributos.iloc[:, mejor_atributo].astype(float) <= punto_corte_2)
                rango_3 = atributos.iloc[:, mejor_atributo].astype(float) > punto_corte_2
                if np.any(rango_1):
                    arbol[f'<= {punto_corte_1}'] = self._construir_arbol(atributos[rango_1], etiqueta[rango_1], max_range_split, categoricos)
                if np.any(rango_2):
                    arbol[f'> {punto_corte_1} y <= {punto_corte_2}'] = self._construir_arbol(atributos[rango_2], etiqueta[rango_2], max_range_split, categoricos)
                if np.any(rango_3):
                    arbol[f'> {punto_corte_2}'] = self._construir_arbol(atributos[rango_3], etiqueta[rango_3], max_range_split, categoricos)
        
        return arbol
    
    def predict(self, x):
        nodo = self.arbol
        while isinstance(nodo, dict):
            atributo = nodo['atributo']
            puntos_corte = nodo['puntos_corte']
            
            if puntos_corte is None:
                valor = x[atributo]
                if valor in nodo:
                    nodo = nodo[valor]
                else:
                    return None  # Valor no esperado
            else:
                if len(puntos_corte) == 1:
                    punto_corte = puntos_corte[0]
                    if x[atributo] <= punto_corte:
                        nodo = nodo.get(f'<= {punto_corte}', None)
                    else:
                        nodo = nodo.get(f'> {punto_corte}', None)
                elif len(puntos_corte) == 2:
                    punto_corte_1 = puntos_corte[0]
                    punto_corte_2 = puntos_corte[1]
                    if x[atributo] <= punto_corte_1:
                        nodo = nodo.get(f'<= {punto_corte_1}', None)
                    elif x[atributo] <= punto_corte_2:
                        nodo = nodo.get(f'> {punto_corte_1} y <= {punto_corte_2}', None)
                    else:
                        nodo = nodo.get(f'> {punto_corte_2}', None)
                
        return nodo

if __name__ == '__main__':
    # Leer el archivo
    DATASET_FILE = 'lab1_dataset.csv'
    dataset = pd.read_csv(DATASET_FILE, sep=",").add_prefix("c")

    # Se elimina del dataset la primera columna ya que no es un atributo. Corresponde al ID del paciente
    dataset = dataset.drop(dataset.columns[0], axis=1)

    categorical_columns = [1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]

    atributos = dataset.iloc[:, 1:].copy()
    etiqueta = dataset.iloc[:, 0].copy()

    # Se convierten los valores numericos utilizando discretizacion
    atributos_max_range_split_2 = atributos.copy()
    atributos_max_range_split_3 = atributos.copy()

    for col in range(atributos.shape[1]):
        if col not in categorical_columns:
            punto_corte_2 = _encontrar_mejores_puntos_corte(atributos_max_range_split_2, atributos.iloc[:, col], col, etiqueta, 2)
            atributos_max_range_split_2.iloc[:, col] = pd.cut(atributos.iloc[:, col], bins=[-np.inf] + punto_corte_2 + [np.inf], labels=range(len(punto_corte_2)+1)).astype(int)
            
            puntos_corte_3 = _encontrar_mejores_puntos_corte(atributos_max_range_split_3, atributos.iloc[:, col], col, etiqueta, 3)
            atributos_max_range_split_3.iloc[:, col] = pd.cut(atributos.iloc[:, col], bins=[-np.inf] + puntos_corte_3 + [np.inf], labels=range(len(puntos_corte_3)+1)).astype(int)

    # Aquí se definen correctamente las variables dataset_max_2 y dataset_max_3
    dataset_max_2 = pd.concat([etiqueta, atributos_max_range_split_2], axis=1)
    dataset_max_3 = pd.concat([etiqueta, atributos_max_range_split_3], axis=1)

    # Preparar validación cruzada de 5 pliegues
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Realizar validación cruzada para max_range_split = 2
    precision_m2 = []
    for train_index, test_index in kf.split(dataset_max_2):
        train_2 = dataset_max_2.iloc[train_index]
        test_2 = dataset_max_2.iloc[test_index]

        atributos_2 = train_2.iloc[:, 1:]
        etiqueta_2 = train_2.iloc[:, 0]
        atributos_test_2 = test_2.iloc[:, 1:]
        etiqueta_test_2 = test_2.iloc[:, 0]

        arbol_max_range_split_2 = ArbolDecision()
        arbol_max_range_split_2.fit(atributos_2, etiqueta_2, 2, categorical_columns)

        predicciones_m2_test = [arbol_max_range_split_2.predict(x) for x in atributos_test_2.values]
        precision_m2.append(np.sum(np.array(predicciones_m2_test) == etiqueta_test_2.values) / len(etiqueta_test_2))

    print("Algoritmo ID3 con max_iter_split = 2:")
    print(f"Precisión promedio: {np.mean(precision_m2) * 100}%")
    print(f"Desviación estándar: {np.std(precision_m2) * 100}%")
    print('\n')

    # Realizar validación cruzada para max_range_split = 3
    precision_m3 = []
    for train_index, test_index in kf.split(dataset_max_3):
        train_3 = dataset_max_3.iloc[train_index]
        test_3 = dataset_max_3.iloc[test_index]

        atributos_3 = train_3.iloc[:, 1:]
        etiqueta_3 = train_3.iloc[:, 0]
        atributos_test_3 = test_3.iloc[:, 1:]
        etiqueta_test_3 = test_3.iloc[:, 0]

        arbol_max_range_split_3 = ArbolDecision()
        arbol_max_range_split_3.fit(atributos_3, etiqueta_3, 3, categorical_columns)

        predicciones_m3_test = [arbol_max_range_split_3.predict(x) for x in atributos_test_3.values]
        precision_m3.append(np.sum(np.array(predicciones_m3_test) == etiqueta_test_3.values) / len(etiqueta_test_3))

    print("Algoritmo ID3 con max_iter_split = 3:")
    print(f"Precisión promedio: {np.mean(precision_m3) * 100}%")
    print(f"Desviación estándar: {np.std(precision_m3) * 100}%")
    print('\n')