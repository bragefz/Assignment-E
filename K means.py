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
from sklearn.cluster import KMeans
from sklearn import metrics

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
model = KMeans(n_clusters=6, init="k-means++", n_init=15)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# testing model
acc = metrics.accuracy_score(y_test, y_pred)

print("Model accuracy: %.3f" % acc)