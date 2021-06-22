from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

petals = load_iris()
# print(dir(petals))
# ['DESCR', 'data', 'feature_names', 'filename', 'frame', 'target', 'target_names']

# print(petals.data)
my_new_data = pd.DataFrame(petals.data)
# print(my_new_data.head())
# print(petals.target)
my_new_data['target'] = petals.target
# print(my_new_data)
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test = train_test_split(my_new_data.drop(['target'],1),petals.target,test_size=0.1)

model = RandomForestClassifier(n_estimators=30,n_jobs=-1,oob_score=True,max_features="auto",min_samples_leaf=5)
model.fit(X_train,Y_train)
acc = model.score(x_test,y_test)
print(acc)
# 0.8666666666666667 my first accuracy 0.9333333333333333 1.0

from sklearn.metrics import confusion_matrix

#confusion matrix
y_pred = model.predict(x_test)
# print(y_pred)

conm = confusion_matrix(y_test,y_pred)
print(conm)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.hist(conm)
plt.xlabel("predicted value")
plt.ylabel("Truth of prediction")
plt.title("Iris dataset prediction graph")
plt.show()