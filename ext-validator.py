import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Definir el path para los archivos CSV
path = '/mnt/data/'
files = glob.glob(os.path.join(path, 'hog_*.csv'))

# Leer y almacenar los datasets en un diccionario
datasets = {os.path.basename(f): pd.read_csv(f) for f in files}

# Definir el modelo KNN y el espacio de búsqueda para RandomizedSearchCV
knn = KNeighborsClassifier()
param_dist = {
    'n_neighbors': np.arange(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Evaluar cada dataset utilizando validación cruzada con RandomizedSearchCV
results = {}
for name, data in datasets.items():
    X = data.iloc[:, :-1]  # Asumiendo que las características están en todas las columnas excepto la última
    y = data.iloc[:, -1]   # Asumiendo que la última columna es la etiqueta
    random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    best_score = random_search.best_score_
    results[name] = best_score

# Ordenar los resultados y seleccionar el top 3
sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
top_3_datasets = sorted_results[:3]

# Mostrar el top 3
print("Top 3 Datasets:")
for dataset, score in top_3_datasets:
    print(f"{dataset}: {score}")

# Graficar los scores
names = [item[0] for item in sorted_results]
scores = [item[1] for item in sorted_results]

plt.figure(figsize=(10, 6))
plt.barh(names, scores, color='skyblue')
plt.xlabel('Validation Score')
plt.title('Comparison of Datasets by Validation Score')
plt.gca().invert_yaxis()
plt.show()
