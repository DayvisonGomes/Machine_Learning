import numpy as np
import matplotlib.pyplot as plt

def zeros(linhas, colunas):
    return np.zeros((linhas,colunas))

def ones(linhas, colunas):
    return np.ones((linhas,colunas))

def random_normal(linhas , colunas):
    return np.random.randn(linhas , colunas)

def random_uniform(rows , cols):
    return np.random.rand(rows , cols)

def glorot_normal(rows , cols):
    std_dev = np.sqrt(2.0 / (rows , cols))
    return std_dev * np.random.randn(rows , cols)

class LinearRegression():

    def __init__(self, dim_entrada, learning_rate=1e-3, ini_pesos=random_normal):
        self.learning_rate = learning_rate
        self.w = ini_pesos(1,dim_entrada)
        self.b = np.random.uniform(-1,1)
    
    def fit(self, X_train, y_train, num_iterations = 1000, verbose=100):

        for iteration in range(num_iterations + 1):

            dw,db,cost = self.__feedforward(X_train, y_train)

            self.w = self.w + 1e-7*dw
            self.b = self.b + self.learning_rate*db

            if iteration % verbose == 0:
                print('epoch: {0:=4}/{1} loss_train: {2:.8f}'.format(iteration,num_iterations,cost))

    def predict(self, X_test):
        
        w,b = self.get_w_b()

        return np.array(X_test*w + b)

    def __feedforward(self, X_train, y_train):

        y_pred = np.dot(X_train, self.w.T) + self.b
        
        error = np.array(y_train - y_pred)
        cost = 0.5*np.mean( (error)**2 )                            
        
        dw = np.dot(error.T, X_train)
        db = error.sum()

        return dw,db,cost

    def get_w_b(self):
        return self.w, self.b


x = np.array([-5,-3,-1,0,2,5,6,9]).reshape(-1,1)
y = np.array([-3,-1,0,3,4,6,8,10]).ravel()
dim = x.shape[1]

model = LinearRegression(dim, learning_rate=1e-2, ini_pesos=random_uniform)
model.fit(x,y,num_iterations=3000,verbose=500)

plt.figure(figsize=(12,6))
plt.grid()
plt.scatter(x,y, color='black')
plt.plot(x, model.predict(x), color='red')
plt.show()
