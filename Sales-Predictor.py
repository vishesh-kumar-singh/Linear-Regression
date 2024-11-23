import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gradient Descent Training Function
def model(A,l_rate,n,x_train,y_train):

    m=x_train.shape[1]
    for i in range(n):

        error=(np.dot(A,x_train)-y_train)
        gradient=(2/m)*np.dot(error,x_train.T)

        NewA=A-l_rate*gradient
        A=NewA
        

    return A

def rmse(A, x_train, y_train):
    predictions = np.dot(A, x_train)  # Compute predictions for all samples
    errors = predictions - y_train  # Compute the residuals
    return (np.mean(errors ** 2))**(1/2)  # Root Mean squared error


training=pd.read_csv("Dummy Data HSS.csv")
training.dropna(inplace=True)
x_array=training[["TV","Radio","Social Media"]].values.T
new_row=np.ones((1,x_array.shape[1]))
x_train=np.vstack([new_row,x_array])

y_train=training[["Sales"]].values.T


A=np.zeros((x_train.shape[0]))

Output_Matrix=model(A,0.0001,70000,x_train,y_train)[0].tolist()
intercept=Output_Matrix[0]
slopet=Output_Matrix[1]
sloper=Output_Matrix[2]
slopes=Output_Matrix[3]

final_A=np.array(Output_Matrix)

error=rmse(final_A,x_train,y_train)

user_budgett=float(input("What are you planning to invest in Television Advertisement in millions: "))
user_budgetr=float(input("What are you planning to invest in Radio Advertisement in millions: "))
user_budgets=float(input("What are you planning to invest in Social Media Advertisement in millions: "))
sales=intercept+slopes*user_budgets+slopet*user_budgett+sloper*user_budgetr
confidence_99=error*2.576
confident99_sales_lower=sales-confidence_99
confident99_sales_upper=sales+confidence_99
confidence_95=error*1.96
confident95_sales_lower=sales-confidence_95
confident95_sales_upper=sales+confidence_95
confidence_90=error*1.645
confident90_sales_lower=sales-confidence_90
confident90_sales_upper=sales+confidence_90
print(f"Your expected revenue with 90% confidence is between {confident90_sales_lower} Millions to {confident90_sales_upper} Millions.")
print(f"Your expected revenue with 95% confidence is between {confident95_sales_lower} Millions to {confident95_sales_upper} Millions.")
print(f"Your expected revenue with 99% confidence is between {confident99_sales_lower} Millions to {confident99_sales_upper} Millions.")