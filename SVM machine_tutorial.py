from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.datasets import load_diabetes,load_breast_cancer
from sklearn import svm,metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=.1)
classes = ['malignant', 'benign']
my_model = svm.SVC(kernel="linear",C=2)
# my_model = KNeighborsClassifier(n_neighbors=13)#0.9298245614035088
my_model.fit(x_train,y_train)
#0.9298245614035088
y_pred = my_model.predict(x_test)

acc = metrics.accuracy_score(y_test,y_pred)
print(acc) #0.9824561403508771