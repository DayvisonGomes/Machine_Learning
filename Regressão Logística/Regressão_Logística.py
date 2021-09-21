import numpy as np

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


class LogisticRegression():

    def __init__(self, dim_entrada, learning_rate = 1e-3, ini_pesos=random_normal):
        self.learning_rate = learning_rate
        self.w = ini_pesos(1,dim_entrada)
        self.b = np.random.uniform(-1,1)
    
    def sigmoid(self, x):

        return 1/(1 + np.exp(-x))
    
    def fit(self, X_train, y_train, num_iterations = 1000, verbose=100):

        for iteration in range(num_iterations + 1):

            dw,db,cost = self.__feedforward(X_train, y_train)

            self.w = self.w + self.learning_rate*dw
            self.b = self.b + self.learning_rate*db

            if iteration % verbose == 0:
                print('epoch: {0:=4}/{1} loss_train: {2:.8f}'.format(iteration,num_iterations,cost))

    def predict(self, X_test):

        y_pred = [1 if self.sigmoid(np.dot(x, self.w.T) + self.b) >= 0.5 else 0 for x in X_test]

        return np.array(y_pred) 

    def __feedforward(self, X_train, y_train):

        z = np.dot(X_train, self.w.T) + self.b
        y_pred = self.sigmoid(z)

        cost = np.mean(-y_train*np.log(y_pred) - (1-y_train)*np.log(1-y_pred))                                
        error = y_train - y_pred
        
        dw = np.dot(error.T, X_train)
        db = error.sum()

        return dw,db,cost

    def get_w_b(self):
        return self.w, self.b
