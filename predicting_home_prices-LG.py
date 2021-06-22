import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import linear_model

data = pd.read_csv("homeprices.csv")

# print(data.head())

model = linear_model.LinearRegression()

model.fit(data[['area']],data.price)
y = model.predict([[3300]])
# print(y)

areas_data = pd.read_csv('areas.csv')
area_pred = model.predict(areas_data)
# print(area_pred)
areas_data["Actual Price"] = area_pred
print(areas_data.head(15))
areas_data.to_csv("Areas+predicted_values.csv",index=False)

plt.xlabel("area",fontsize=20)
plt.ylabel("price",fontsize=20)
plt.scatter(data.area,data["price"],color="red",marker="+",label="actual data")
plt.plot(data.area,model.predict(data[['area']]),color="blue",label="area prediction")
plt.legend()
plt.show()

