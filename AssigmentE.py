Ignore this file

"""""
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# selecting which features to include
data = data[["Diabetes_binary", "HighBP", "HighChol", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
             "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk", "Sex", "Age", "Education", "Income"]]

print(data.head())

# choosing which label for algorithm to predict
target = "Diabetes_binary"

# creating one array without the target label and one with only target label
w = np.array(data.drop([target], axis=1))
t = np.array(data[target])

# splitting the data into training and testing sets
w_train, w_test, t_train, t_test = sklearn.model_selection.train_test_split(w, t, test_size=0.2)
#sklearn.
linear = linear_model.LinearRegression()

linear.fit(w_train, t_train)
acc = linear.score(w_test, t_test)
#print(acc)

#print("Coefficient: \n", linear.coef_)
#print("Intercept \n", linear.intercept_)


"""Remove:
cholcheck
physactivity
anyhealthcare
nodocbccost
genhlth
menthlth
physhlth
"""
""""""