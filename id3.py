import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd

def calcular_entropia(etiqueta):
    valores, conteos = np.unique(etiqueta, return_counts=True)
    probabilidades = conteos / len(etiqueta)
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    return entropia

def calcular_ganancia(atributos, etiqueta, atributo_a_calcular, puntos_corte):
    entropia_inicial = calcular_entropia(etiqueta)
    
    if puntos_corte is None or len(puntos_corte) == 0:
        valores, conteos = np.unique(atributos[:, atributo_a_calcular], return_counts=True)
        entropia_condicional = sum((conteos[i] / sum(conteos)) * calcular_entropia(etiqueta[atributos[:, atributo_a_calcular] == v]) for i, v in enumerate(valores))
    
    # Si hay un solo punto de corte
    elif len(puntos_corte) == 1:
        punto_corte = puntos_corte[0]
        menores = atributos[:, atributo_a_calcular].astype(float) <= punto_corte
        mayores = atributos[:, atributo_a_calcular].astype(float) > punto_corte
        
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
        rango_1 = atributos[:, atributo_a_calcular].astype(float) <= punto_corte_1
        rango_2 = (atributos[:, atributo_a_calcular].astype(float) > punto_corte_1) & (atributos[:, atributo_a_calcular].astype(float) <= punto_corte_2)
        rango_3 = atributos[:, atributo_a_calcular].astype(float) > punto_corte_2
        
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

def _es_binario(atributo):
    return len(np.unique(atributo)) <= 2

def _encontrar_mejores_puntos_corte(atributo, etiqueta, max_range_split):
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
        puntos_corte = puntos_corte[:max_range_split]
    
    return puntos_corte

class ArbolDecision:
    def __init__(self):
        self.arbol = None
    
    def fit(self, atributos, etiqueta, max_range_split):
        self.arbol = self._construir_arbol(atributos, etiqueta, max_range_split)
    
    def _construir_arbol(self, atributos, etiqueta, max_range_split):
        if len(np.unique(etiqueta)) == 1:
            return np.unique(etiqueta)[0]
        
        if atributos.shape[1] == 0 or len(atributos) == 0:
            return np.bincount(etiqueta).argmax()
        
        mejor_atributo = None
        mejor_ganancia = -1
        mejor_puntos_corte = None
        
        for i in range(atributos.shape[1]):
            if _es_binario(atributos[:, i]):
                ganancia = calcular_ganancia(atributos, etiqueta, i, None)
                puntos_corte = None
            else:
                puntos_corte = _encontrar_mejores_puntos_corte(atributos[:, i], etiqueta, max_range_split)
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
            valores = np.unique(atributos[:, mejor_atributo])
            for valor in valores:
                indices = atributos[:, mejor_atributo] == valor
                if len(indices) > 0:
                    arbol[valor] = self._construir_arbol(atributos[indices], etiqueta[indices], max_range_split)
        else:
            if len(mejor_puntos_corte) == 1:
                punto_corte = mejor_puntos_corte[0]
                menores = atributos[:, mejor_atributo].astype(float) <= punto_corte
                mayores = atributos[:, mejor_atributo].astype(float) > punto_corte
                if np.any(menores):
                    arbol[f'<= {punto_corte}'] = self._construir_arbol(atributos[menores], etiqueta[menores], max_range_split)
                if np.any(mayores):
                    arbol[f'> {punto_corte}'] = self._construir_arbol(atributos[mayores], etiqueta[mayores], max_range_split)
            elif len(mejor_puntos_corte) == 2:
                punto_corte_1 = mejor_puntos_corte[0]
                punto_corte_2 = mejor_puntos_corte[1]
                rango_1 = atributos[:, mejor_atributo].astype(float) <= punto_corte_1
                rango_2 = (atributos[:, mejor_atributo].astype(float) > punto_corte_1) & (atributos[:, mejor_atributo].astype(float) <= punto_corte_2)
                rango_3 = atributos[:, mejor_atributo].astype(float) > punto_corte_2
                if np.any(rango_1):
                    arbol[f'<= {punto_corte_1}'] = self._construir_arbol(atributos[rango_1], etiqueta[rango_1], max_range_split)
                if np.any(rango_2):
                    arbol[f'> {punto_corte_1} y <= {punto_corte_2}'] = self._construir_arbol(atributos[rango_2], etiqueta[rango_2], max_range_split)
                if np.any(rango_3):
                    arbol[f'> {punto_corte_2}'] = self._construir_arbol(atributos[rango_3], etiqueta[rango_3], max_range_split)
        
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

# Leer el archivo y dividir en conjunto de entrenamiento y prueba
DATASET_FILE = 'lab1_dataset.csv'
dataset = pd.read_csv(DATASET_FILE, sep=",", header=None).add_prefix("c")
dataset = dataset.drop(dataset.columns[0], axis=1)

# Convertir todos los atributos a valores numéricos
le = preprocessing.LabelEncoder()
for column in dataset.columns:
    dataset[column] = le.fit_transform(dataset[column])

train, test = model_selection.train_test_split(dataset, test_size=0.2, random_state=42)

# Entrenamiento del modelo
atributos = train.iloc[:, 1:].values
etiqueta = train.iloc[:, 0].values

arbol = ArbolDecision()
arbol.fit(atributos, etiqueta, 3)

# Validar el árbol con el conjunto de prueba
atributos_test = test.iloc[:, 1:].values
etiqueta_test = test.iloc[:, 0].values

# Hacer predicciones y calcular la precisión
predicciones = [arbol.predict(x) for x in atributos_test]
precision = np.sum(np.array(predicciones) == etiqueta_test) / len(etiqueta_test)
print(f"Precisión: {precision}")