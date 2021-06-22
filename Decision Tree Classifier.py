from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("salaries.csv")

# print(data.head(10))
working_data = data.drop(["salary_more_then_100k"],1)
predict_data = data["salary_more_then_100k"]

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

working_data["up_company"] = le_company.fit_transform(working_data["company"])
working_data["up_job"] = le_job.fit_transform(working_data["job"])
working_data["ip_degree"] = le_degree.fit_transform(working_data["degree"])

#dropping columns
working_data.drop(["company","job","degree"],1,inplace=True)
# print(working_data.head())

model = DecisionTreeClassifier()
model.fit(working_data,predict_data)
score = model.score(working_data,predict_data)
print("The score accuracy is : ",score) #1.0

pred = model.predict([[2,1,1]])
print(pred)

