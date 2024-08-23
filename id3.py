

def calcular_entropia(atributos, etiqueta, atributo_a_calcular):
    pass

def calcular_ganancia(atributos, etiqueta, atributo_a_calcular):
    pass
    


class ArbolDecision:
    def __init__(self):
        self.arbol = None
    
    def fit(self, atributos, etiqueta):
        self.arbol = self._construir_arbol(atributos, etiqueta)
    
    def _construir_arbol(self, atributos, etiqueta, depth=0):
        # Si todas las etiquetas son iguales, devuelve la etiqueta
        if len(np.unique(etiqueta)) == 1:
            return etiqueta[0]
        
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

            subtree = self._construir_arbol(subset_X, subset_y, depth + 1)
            tree[mejor_atributo][v] = subtree
        
        return tree
    




