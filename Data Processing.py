import seaborn as sns
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn import linear_model
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
# attaching features and classes together to one dataset
data = X
data.insert(0, "Diabetes_012", y)


# GRAPHS AND PLOTS

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

# correlation matrix
# print(data_imbalanced.corr())

# unbalanced data pie plot
#plt.pie(y_imba.value_counts(), labels=["No Diabetes", "Diabetes", "Prediabetes"], autopct="%1.1f%%")
#plt.show()

data = X
data.insert(0, "Diabetes_012", y)

# finding mean
mean = data[["Diabetes_012", "HighBP", "HighChol", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "Fruits", "Veggies", "HvyAlcoholConsump",
             "DiffWalk", "Sex", "Age", "Education", "Income"]].mean()

# finding mode
mode = data[["Diabetes_012", "HighBP", "HighChol", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "Fruits", "Veggies", "HvyAlcoholConsump",
             "DiffWalk", "Sex", "Age", "Education", "Income"]].mode()

# finding median
median = data[["Diabetes_012", "HighBP", "HighChol", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "Fruits", "Veggies", "HvyAlcoholConsump",
             "DiffWalk", "Sex", "Age", "Education", "Income"]].median()

# finding standard deviation
std = data.std()

# finding q1
q1 = data.quantile(q=0.25)

# finding q3
q3 = data.quantile(q=0.75)

# making pairplot to see effect on "Diabetes_012" by different features
sns.pairplot(data, hue="Diabetes_012", y_vars=["Age"],
             x_vars=["Age"], kind="reg")


plt.show()

#print(data.head())


