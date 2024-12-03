# Using Data Normalisation

This code represents implementation of data normalisation in my **Linear Regression Algorithm**.

---


## How the Code Works 🛠️

The training data is normalised by subtracting the minimum value of each column and then divided by the range of the data column to make the value between 0 to 1. The normalisation is helpfull if different features have different range of values, without normalisation in such cases one of the feature may contribute more to the model.

Then for predictions the range and minimum value of test data is used to normalise it then the output is de-normalised giving the predictions.