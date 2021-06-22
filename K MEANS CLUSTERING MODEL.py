from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

data = pd.read_csv("income.csv")
# plot.scatter(data['Age'],data['Income($)'])
# plot.show()

# print(data.head())




# data["cluster"] = y_pred
# print(data.head())

my_scaler = MinMaxScaler()
my_scaler.fit(data[["Income($)"]])
data["Income($)"] = my_scaler.transform(data[["Income($)"]])
my_scaler.fit(data[['Age']])
data.Age = my_scaler.transform(data[["Age"]])
# print(data.head())
model = KMeans(n_clusters=3,n_jobs=-1)
y_pred = model.fit_predict(data[['Age','Income($)']])
data["Cluster"] = y_pred
# print(y_pred)

# print(model.cluster_centers_)

#elbow method
k = range(1,10)
sse = []
for _ in k:
    km = KMeans(n_clusters=_)
    km.fit(data[["Age","Income($)"]])
    sse.append(km.inertia_)
print(sse)

plot.xlabel('K')
plot.ylabel('Sum of squared error!')
plot.plot(k,sse)
plot.show()
