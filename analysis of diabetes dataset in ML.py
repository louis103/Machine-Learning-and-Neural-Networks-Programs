import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#loading data
data = pd.read_csv("diabetes.csv")
# print(data.head())
"""Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')"""
training_data = data.drop(["Outcome"],1)
main_target = data.Outcome

#splitting data into train and test
X_train,X_test,y_train,y_test = train_test_split(training_data,main_target,test_size=0.1,random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

acc = model.score(X_test,y_test)
print("Model Accuracy == ",acc*100,"%")
chances = ["Not likely to have Diabetes","Likely to have Diabetes"]
#my variables
Pregnancies = 3
Glucose = 190
BloodPressure = 86
SkinThickness = 24
Insulin = 500
BMI = 28.6
DiabetesPedigreeFunction = 0.351
Age = 55


predictions = model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
print(predictions)
print(f"The person is : {chances[int(predictions)]}")
