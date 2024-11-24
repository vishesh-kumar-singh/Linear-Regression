import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gradient Descent Training Function
def model(x_train,y_train,n=100000,l_rate=0.0001):

    A=np.zeros((x_train.shape[0]))
    m=x_train.shape[1]
    for i in range(n):

        error=(np.dot(A,x_train)-y_train)
        gradient=(2/m)*np.dot(error,x_train.T)

        NewA=A-l_rate*gradient
        A=NewA
        

    return A
# Root Mean Squared Error
def rmse(A, x_train, y_train):
    predictions = np.dot(A, x_train)  
    errors = predictions - y_train
    return (np.mean(errors ** 2))**(1/2)

# Training Data
training=pd.read_csv("Dummy Data HSS.csv")
training.dropna(inplace=True)
x_array=training[["TV","Radio","Social Media"]].values.T
new_row=np.ones((1,x_array.shape[1]))
x_train=np.vstack([new_row,x_array])

y_train=training[["Sales"]].values.T

Output_Matrix=model(x_train,y_train,70000,0.0001)[0]
print(Output_Matrix)

# RMSE
error=rmse(Output_Matrix,x_train,y_train)


# Predicting user's Results
user_budgett,user_budgetr,user_budgets=map(float,input("What are you planning to invest in Television, Radio and Social Media Advertisement respectively in millions: ").split(','))

budget=np.array([1,user_budgett,user_budgetr,user_budgett]).T

sales=np.dot(Output_Matrix,budget)


# Output
confidence_levels = {90: 1.645,95: 1.96,99: 2.576}
confidence_intervals = {}

for level, z_score in confidence_levels.items():
    confidence = error * z_score
    lower_bound = sales - confidence
    upper_bound = sales + confidence
    confidence_intervals[level] = (lower_bound, upper_bound)

for level, (lower, upper) in confidence_intervals.items():
    print(f"Your expected revenue with {level}% confidence is between {lower} Millions to {upper} Millions.")