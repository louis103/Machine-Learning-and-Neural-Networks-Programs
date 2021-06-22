import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("canada_per_capita_income.csv")
minn = MinMaxScaler()
# data['year'] = minn.fit_transform(data[['year']])
# data['per capita income (US$)'] = minn.fit_transform(data[['per capita income (US$)']])
#
#
# print(data.head())

# X_train,x_test,Y_train,y_test = train_test_split(data[["year"]],data["per capita income (US$)"],test_size=0.3)

model = LinearRegression()
# model.fit(X_train,Y_train)
X = data[["year"]]
y = data["per capita income (US$)"]

model.fit(X,y)
plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.show()

score = model.score(X,y)
print("The accuracy is : ",score)

pred = model.predict([[1980]])
print("Predicted is: ",pred)


# ans = 41122.11676017


