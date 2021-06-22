import sklearn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression

data = pd.read_csv('heart.csv')

X_data = data.drop(['output'],1)
Y_data = data.output

X_train,x_test,Y_train,y_test = train_test_split(X_data,Y_data,test_size=0.3,random_state=4)


model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
acc = model.score(x_test,y_test)
print(acc)

predictions = model.predict([[67,1,0,160,286,0,0,108,1,1.5,1,3,2]])
print(predictions)