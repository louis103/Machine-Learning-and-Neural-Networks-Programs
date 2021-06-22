from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv',sep=";")
# print(data)
data = data[["G1","G2","G3","studytime","failures","absences"]]
# print(data.head())
predict_value = "G3"
X = np.array(data.drop([predict_value], 1))
y = np.array(data[predict_value])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
best_model = 0
"""
for _ in range(30):
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    linear = LinearRegression()
    linear.fit(x_train,y_train)
    #
    accuracy = linear.score(x_test,y_test)
    print(accuracy)
    if accuracy>best_model:
        best_model = accuracy
        with open('studentmodel.pickle','wb') as f:
            pickle.dump(linear,f)
            
"""

pickle_in = open('studentmodel.pickle','rb')
linear = pickle.load(pickle_in)
print("Coefficients: ",linear.coef_)
print("Intercept: ",linear.intercept_)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])
p = "studytime"
style.use('ggplot')
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Predicted grade")
plt.show()
