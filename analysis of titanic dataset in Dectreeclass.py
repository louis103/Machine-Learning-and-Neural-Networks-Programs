from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("titanic.csv")
data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
# print(data.head())

train_data = data.drop(["Survived"],1)
target_input = data.Survived

train_data.Sex = train_data.Sex.map({'male': 1, 'female': 2})

# print(train_data.Age[:10])

mean_v = train_data.Age.mean()
train_data.Age = train_data["Age"].fillna(mean_v)
# print(train_data.Age[:10])

X_train,X_test,y_train,y_test = train_test_split(train_data,target_input,test_size=0.1)
print(len(X_train),len(X_test))

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

acc = model.score(X_test,y_test)
print("Model Accuracy == ",acc*100,"%")
survival = ["Did not survive","Survived"]
# print(train_data.head())
predicted = model.predict([[3,1,22,7.25]])
print("Predicted value == ",predicted)
for i in range(len(predicted)):
    print("chances are == ",survival[predicted[i]])
#    Pclass  Sex   Age     Fare
# 0       3    1  22.0   7.2500
# 1       1    2  38.0  71.2833
# 2       3    2  26.0   7.9250
# 3       1    2  35.0  53.1000
# 4       3    1  35.0   8.0500
