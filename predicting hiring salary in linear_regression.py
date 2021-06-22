import pandas as pd

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("hiring.csv")
# print(data.head())
d1 = LabelEncoder()

data["experience"] = data.experience.fillna("0")
# print(data.head())
data["experience"] = d1.fit_transform(data["experience"])

mean = data["test_score(out of 10)"].mean()
data["test_score(out of 10)"] = data["test_score(out of 10)"].fillna(mean)
# print(data.head(10))

model = LinearRegression()
model.fit(data[["experience","test_score(out of 10)","interview_score(out of 10)"]],data["salary($)"])
# score = model.score(data[["experience","test_score(out of 10)","interview_score(out of 10)"]],data["salary($)"])
# print("The acc is : ",score)

pred = model.predict([[2,9,6]])
print("Predicted salary($) is: ",pred)