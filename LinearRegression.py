import numpy as np


class LinearRegression:
    def __init__(self,n=100000,lr=0.0001):
        self.lr=lr
        self.n=n
        self.A=None
        self.training_rmse=None

    def fit(self,x_train,y_train):
        new_row=np.ones((x_train.shape[0],1))
        x_train_with_bias=np.hstack([new_row,x_train])

        self.A=np.zeros((x_train_with_bias.shape[1],1))
        m=x_train.shape[0]
        for i in range(self.n):

            error=(np.dot(x_train_with_bias,self.A)-y_train)
            gradient=(2/m)*np.dot(x_train_with_bias.T,error)

            self.A=self.A-self.lr*gradient
        predictions = np.dot(x_train_with_bias,self.A)  
        errors = predictions - y_train
        self.training_rmse=(np.mean(errors ** 2))**(1/2)

    def predict(self,X):
        new_row=np.ones((X.shape[0],1))
        X_final=np.hstack([new_row,X])
        return np.dot(X_final,self.A)
    
    def test(self,x_test,y_test):
        new_row=np.ones((x_test.shape[0],1))
        X_final=np.hstack([new_row,x_test])
        predictions = np.dot(X_final,self.A)  
        errors = predictions - y_test
        return (np.mean(errors ** 2))**(1/2)