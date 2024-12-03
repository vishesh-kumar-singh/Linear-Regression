import numpy as np
from Normalizer import normalizer


class LinearRegression:
    def __init__(self,n=100000,lr=0.0001):
        self.lr=lr
        self.n=n
        self.A=None
        self.training_rmse=None
        self.y_intercept=None
        self.x_intercept=None
        self.y_multiple=None
        self.x_multiple=None

    def fit(self,x_train,y_train):
        new_row=np.ones((x_train.shape[0],1))
        x_train_with_bias=np.hstack([new_row,x_train])

        x_normalizer=normalizer()
        x_normalized=x_normalizer.normalize(x_train_with_bias)
        self.x_intercept=x_normalizer.min
        self.x_multiple=x_normalizer.denominator

        y_normalizer=normalizer()
        y_normalizer.normalize(y_train)
        y_normalized=y_normalizer.normalize(y_train)
        self.y_multiple=y_normalizer.denominator
        self.y_intercept=y_normalizer.min

        self.A=np.zeros((x_normalized.shape[1],1))
        m=x_train.shape[0]
        # print(y_normalized)
        for i in range(self.n):

            error=(np.dot(x_normalized,self.A)-y_normalized)
            gradient=(2/m)*np.dot(x_normalized.T,error)

            self.A=self.A-self.lr*gradient
        normalized_predictions = np.dot(x_normalized,self.A)
        print("\n",self.A,"\n")
        # print(normalized_predictions)
        predictions=(normalized_predictions*self.y_multiple)+self.y_intercept
        # print(predictions)
        errors = predictions - y_train
        self.training_rmse=(np.mean(errors ** 2))**(1/2)
        # print(self.training_rmse)

    def predict(self,X):
        new_row=np.ones((X.shape[0],1))
        X_final=np.hstack([new_row,X])
        x_normalizer=normalizer()
        x_normalized=x_normalizer.spnormalize(X_final,self.x_multiple,self.x_intercept)
        output=np.dot(x_normalized,self.A)
        return ((output*self.y_multiple)+self.y_intercept)
    
    def test(self,x_test,y_test):
        new_row=np.ones((x_test.shape[0],1))
        X_final=np.hstack([new_row,x_test])
        x_normalizer=normalizer()
        x_normalized=x_normalizer.spnormalize(X_final,self.x_multiple,self.x_intercept)
        output = np.dot(x_normalized,self.A)
        predictions=(output*self.y_multiple)+self.y_intercept
        errors = predictions - y_test
        return (np.mean(errors ** 2))**(1/2)