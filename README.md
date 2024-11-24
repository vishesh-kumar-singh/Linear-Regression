# My First Linear Regression Algorithm 🚀

This code represents my first implementation of a **Linear Regression Algorithm** using **Gradient Descent**. The goal is to predict sales revenue based on investments in three advertising channels: **Television, Radio, and Social Media**. It calculates expected sales and provides confidence intervals (90%, 95%, and 99%) for the prediction.

---

## Features of the Code ✨

1. **Data Preprocessing:**
   - The input data is read from a CSV file named `Dummy Data HSS.csv`.
   - Missing data is removed to ensure a clean dataset.
   - A bias term (intercept) is added as an additional feature to the dataset.

2. **Gradient Descent Implementation:**
   - The weights of the model are updated iteratively to minimize the Mean Squared Error (MSE).
   - The learning rate and the number of iterations are customizable, enabling fine-tuning.

3. **Root Mean Squared Error (RMSE):**
   - RMSE is calculated to evaluate the performance of the model and measure prediction error.

4. **Confidence Intervals:**
   - The code predicts sales revenue for given investments in **TV**, **Radio**, and **Social Media**.
   - It provides **90%**, **95%**, and **99% confidence intervals**, reflecting the range within which the true sales value is likely to lie.

---

## How the Code Works 🛠️

### Step 1: **Data Preprocessing**
- The input data is loaded from a CSV file (`Dummy Data HSS.csv`) containing information about investments in advertising channels (TV, Radio, Social Media) and the corresponding sales revenue.
- Any missing values in the dataset are removed to ensure data quality.
- A bias term (a row of ones) is added to the dataset to include the intercept in the model.

---

### Step 2: **Gradient Descent Implementation**
- The `model` function is used to implement **Gradient Descent**, an optimization algorithm.
- The function iteratively updates the model's weights to minimize the prediction error (difference between actual and predicted sales).
- Each iteration calculates the **gradient**, which is the partial derivative of the error with respect to each weight, and adjusts the weights using the **learning rate**.

---

### Step 3: **Error Calculation**
- The **Root Mean Squared Error (RMSE)** function measures how well the model fits the training data.
- It calculates the square root of the mean of squared residuals (differences between actual and predicted sales).

---

### Step 4: **Model Training**
- The model starts with initial weights of zero and is trained using 70,000 iterations of Gradient Descent.
- After training, the final weights include:
  - **Intercept**: The base sales revenue without any advertising.
  - **Slopes**: The contribution of investments in TV, Radio, and Social Media to sales revenue.

---

### Step 5: **User Interaction**
- The user provides their planned investment amounts for TV, Radio, and Social Media advertisements.
- The code uses the trained model to predict the expected sales revenue based on these inputs.

---

### Step 6: **Confidence Intervals**
- The RMSE is used to compute **confidence intervals**, which indicate the range within which the true sales value is likely to lie for different confidence levels:
  - **90% Confidence Interval**: Calculated using a Z-score of 1.645.
  - **95% Confidence Interval**: Calculated using a Z-score of 1.960.
  - **99% Confidence Interval**: Calculated using a Z-score of 2.576.

---

### Step 7: **Results**
- The code displays the predicted sales revenue and the confidence intervals for 90%, 95%, and 99% confidence levels.

---

## Sample User Interaction 🤖

### User Inputs:
```text
What are you planning to invest in Television, Radio and Social Media Advertisement in millions: 98,37.57,8.52
```
### Output
```text
Your expected revenue with 90% confidence is between [[344.0320446293062]] Millions to [[353.7327452597863]] Millions.
Your expected revenue with 95% confidence is between [[343.1032541434092]] Millions to [[354.66153574568335]] Millions.
Your expected revenue with 99% confidence is between [[341.28695274876605]] Millions to [[356.47783714032647]] Millions.
```

