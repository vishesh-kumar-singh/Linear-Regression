import numpy as np
from Normalizer import normalizer
import pandas as pd


def fit(x_train,y_train):
    new_row=np.ones((x_train.shape[0],1))
    x_train_with_bias=np.hstack([new_row,x_train])
    x_normalizer=normalizer()
    x_normalized=x_normalizer.normalize(x_train_with_bias)
    y_normalizer=normalizer()
    y_normalizer.normalize(y_train)
    y_normalized=y_normalizer.normalize(y_train)
    y_multiple=y_normalizer.denominator
    y_intercept=y_normalizer.min
    print(y_intercept,y_multiple)

training=pd.read_csv("Dummy Data HSS.csv")
training.dropna(inplace=True)
x_train=training[["TV","Radio","Social Media"]].values

y_train=training[["Sales"]].values
fit(x_train,y_train)