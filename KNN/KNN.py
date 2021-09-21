import numpy as np
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def euclidiana(x,y):
    return np.linalg.norm(x-y)

class KNeighborsClassifier_():

    def __init__(self, n_neighbors=3, distancia=euclidiana):
        self.n_neighbors = n_neighbors
        self.distancia = distancia

    def fit_predict(self, X_train, y_train, X_test):
        y_pred = []
        
        for i in range(len(X_test)):
            distancias = [self.distancia(X_test[i], X_train[j]) for j in range(len(X_train))]
            indices = np.argsort(distancias)[:self.n_neighbors].tolist()
            pred = [y_train[indice] for indice in indices] 
        
            dic = Counter(pred)
                
            for key in dic:
                if dic[key] == max(dic.values()):
                    y_pred.append(key)
                    break

        return np.array(y_pred)

X, y = load_digits(return_X_y=True)

X_train , X_test , y_train, y_test = train_test_split(X , y , test_size=0.3, random_state=42)

model = KNeighborsClassifier_(n_neighbors=3)
y_pred = model.fit_predict(X_train, y_train, X_test)

print(f'Acc: {np.round(100*np.mean(y_test==y_pred), 2)}%')

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'Acc: {np.round(100*np.mean(y_test==y_pred), 2)}%')