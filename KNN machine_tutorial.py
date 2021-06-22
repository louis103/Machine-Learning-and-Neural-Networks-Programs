from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model,preprocessing
import matplotlib.pyplot as plt
import pickle

#this is a classifier algorithm

data = pd.read_csv("car.csv")
# print(data.head())
label_encoder = preprocessing.LabelEncoder()
buying = label_encoder.fit_transform(list(data["buying"]))
maint = label_encoder.fit_transform(list(data["maint"]))
door = label_encoder.fit_transform(list(data["door"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
cls = label_encoder.fit_transform(list(data["class"]))
# print(buying)
predict = "class"

X = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)
"""
best = 0
for _ in range(30):
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
    linear = KNeighborsClassifier(n_neighbors=9)
    linear.fit(x_train,y_train)
    accuracy = linear.score(x_test,y_test)
    print(accuracy)
    if accuracy>best:
        best = accuracy
        with open('KNN_Model.pickle','wb') as f:
            pickle.dump(linear,f)
# print(x_train,y_test)
"""
pickle_in = open('KNN_Model.pickle','rb')
model = pickle.load(pickle_in)
# model = KNeighborsClassifier(n_neighbors=9)
# model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)
# print(accuracy)

names = ["unacc","acc","good","vgood"]
predicted = model.predict(x_test)

for i in range(len(x_test)):
    print("Predicted value: ",names[predicted[i]],"Data: ",x_test[i],"Actual value: ",names[y_test[i]])
    kn = model.kneighbors([x_test[i]],9,True)
    print("Neighbors are: ",kn)








