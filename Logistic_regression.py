import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("insurance.csv")
# print(data.head())

# plt.scatter(data.age,data.probability,marker="+",color="red")
# plt.show()
x_train,x_test,y_train,y_test = train_test_split(data[["age"]],data.probability,test_size=0.1)

print(x_test)
model = LogisticRegression()
model.fit(x_train,y_train)
pred_x = model.predict([[40]])
print("prd val: ",pred_x)
y_pred = model.predict(x_test)
print("x_test pred: ",y_pred)
score = model.score(x_test,y_test)
print("score accuracy: ",score)
p = model.predict_proba(x_test)
print("probability",p)