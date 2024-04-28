import seaborn as sns
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import statistics

data_imbalanced = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# selecting which features to include
data_imbalanced = data_imbalanced[["Diabetes_012", "HighBP", "HighChol", "BMI", "Smoker",
                                   "Stroke", "HeartDiseaseorAttack", "Fruits", "Veggies", "HvyAlcoholConsump",
                                   "DiffWalk", "Sex", "Age", "Education", "Income"]]

# choosing which label for algorithm to predict
target = "Diabetes_012"

X_imba = data_imbalanced.drop(["Diabetes_012"], axis=1)
y_imba = data_imbalanced["Diabetes_012"]

# under sampling data to balance it
X, y = RandomUnderSampler().fit_resample(X_imba, y_imba)

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# training model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# testing model
acc = model.score(X_test, y_test)

print("Model accuracy: %.3f" % acc)



