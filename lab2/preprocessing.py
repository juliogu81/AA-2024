from itertools import combinations
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_score
#Prueba chi cuadrado
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

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
        for num_puntos in range(1, max_range_split):
            for combinacion in combinations(puntos_corte, num_puntos):
                combinacion = sorted(combinacion)
                ganancia = calcular_ganancia(atributos, etiqueta, indice, combinacion)
                
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    puntos_corte = combinacion   
    
    return puntos_corte




if __name__ == '__main__':

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

    # Crear un KBinsDiscretizer para discretizar los valores continuos
    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')

    # Aplicar la discretización solo a las columnas continuas
    for col in range(atributos.shape[1]):
        if col not in columnas_categoricas:
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
    

    # Mostrar el dataset discretizado con encabezados
    print(dataset_discretizado_df.head())

    # Definir la variable objetivo
    y = dataset['cid'] 

    # Definir manualmente las columnas categóricas
    columnas_categoricas = ['time','trt','age','wtkg', 'hemo', 'homo', 'drugs','karnof', 'oprior', 'z30', 'zprior','preanti','race', 'gender', 'str2', 'strat', 'symptom', 'treat', 'offtrt','cd40','cd420','cd80','cd820']

    # Seleccionar las columnas categóricas que ya son numéricas
    X_categoricas_encoded = dataset[columnas_categoricas]

    # Aplicar la prueba de Chi-cuadrado
    chi_scores, p_values = chi2(X_categoricas_encoded, y)

    # Mostrar los resultados del Chi-cuadrado para las columnas categóricas
    print("\nPrueba de Chi-cuadrado para columnas categóricas:")
    for i, col in enumerate(columnas_categoricas):
        print(f"Columna: {col}, Chi-cuadrado: {chi_scores[i]}, p-valor: {p_values[i]}")
        
    """ dataset = dataset.drop('zprior', axis=1) """
    print(dataset.head())
        

