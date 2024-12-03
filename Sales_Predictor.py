import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

training=pd.read_csv("Dummy Data HSS.csv")
training.dropna(inplace=True)
x_train=training[["TV","Radio","Social Media"]].values

y_train=training[["Sales"]].values

model=LinearRegression(60000,0.1)
model.fit(x_train,y_train)

user_budgett,user_budgetr,user_budgets=map(float,input("What are you planning to invest in Television, Radio and Social Media Advertisement respectively in millions: ").split(','))

budget=np.array([user_budgett,user_budgetr,user_budgett]).reshape(1,-1)
sales=model.predict(budget)
error=model.training_rmse
confidence_levels = {90: 1.645,95: 1.96,99: 2.576}
confidence_intervals = {}

for level, z_score in confidence_levels.items():
    confidence = error * z_score
    lower_bound = sales - confidence
    upper_bound = sales + confidence
    confidence_intervals[level] = (lower_bound, upper_bound)

for level, (lower, upper) in confidence_intervals.items():
    print(f"Your expected revenue with {level}% confidence is between {lower} Millions to {upper} Millions.")