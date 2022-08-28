# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://user-images.githubusercontent.com/75235090/187084380-3aeba303-c9f7-4be6-9ce5-9534fc9d91a5.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data1 = pd.read_csv('Data22.csv')
data1.head()
X = data1[['input']].values
X
Y = data1[["output"]].values
Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scalar=MinMaxScaler()
scalar.fit(X_train)
scalar.fit(X_test)
X_train=scalar.transform(X_train)
X_test=scalar.transform(X_test)
import tensorflow as tf
model=tf.keras.Sequential([tf.keras.layers.Dense(4,activation='relu'),
                          tf.keras.layers.Dense(5,activation='relu'),
                          tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="rmsprop",metrics=["mse"])
history=model.fit(X_train,Y_train,epochs=1000)
import numpy as np
X_test
preds=model.predict(X_test)
np.round(preds)
tf.round(model.predict([[20]]))
pd.DataFrame(history.history).plot()
r=tf.keras.metrics.RootMeanSquaredError()
r(Y_test,preds)
```
## Dataset Information

![image](https://user-images.githubusercontent.com/75235090/187082589-93401330-68dd-4ee7-9c4e-90b5c04a5ef4.png)


## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/75235090/187083240-58440ebe-5640-443a-aad3-7d1348a72cd5.png)

### Test Data Root Mean Squared Error
![image](https://user-images.githubusercontent.com/75235090/187083277-9f3f172e-727b-4a08-9038-46e3f16a1778.png)


### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/75235090/187083319-97839ffd-8519-4362-bec1-9dccd8b399dd.png)

## RESULT
Thus to develop a neural network model for the given dataset has been implemented successfully.
