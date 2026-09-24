import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
# weights = [1,2,1]
from sklearn import set_config
set_config(transform_output="pandas")


train = pd.read_csv("c:/Users/mishr/Downloads/train.csv")
test = pd.read_csv("c:/Users/mishr/Downloads/test.csv")

x = train.drop(columns='Survived')
y = train['Survived']

# x = x.drop(columns=['Cabin','Name','Ticket', "PassengerId"])
print(y.sample(8))


print(x.head())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# print(x_train.head())


tnf1 = ColumnTransformer([
    ('filing_age', SimpleImputer(), ['Age']),
    ('filing_embarked', SimpleImputer(strategy="most_frequent"), ['Embarked'])
], remainder='passthrough')
print(x_train.head())
tnf2 = ColumnTransformer([(
    'ohe columns',OneHotEncoder(handle_unknown='ignore', sparse_output=False),[3,1]
)],remainder='passthrough')
# print(x_train.head())

tnf3 = StandardScaler()

clf1 = RandomForestClassifier(random_state=42)
clf2 = DecisionTreeClassifier(max_depth=7)
clf3 = KNeighborsClassifier(n_neighbors=3)


estimator = [('rf',clf1), ('dt',clf2), ('knn', clf3)]
for i in range(1,4):
    for j in range(1,4):
        for k in range(1,4):
            vc = VotingClassifier(estimators=estimator, voting="hard",weights=[i, j, k])
            tnf4 = RandomForestClassifier(random_state=42)
            pipe =Pipeline([
                ('tnf1',tnf1),
                ('tnf2',tnf2),
                ('tnf3',tnf3),
                ('tnf4',vc),
            ])
            pipe.fit(x_train,y_train)

            y_pred = pipe.predict(x_test)
            from sklearn.metrics import accuracy_score

            # Get Precision, Recall, and F1-Score for every class at once
            # print("accu",accuracy_score(y_test, y_pred))
            # print(x_train.head())
            cv_scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring='accuracy')

            print("for i={},j={},k{}".format(i,j,k),cv_scores.mean())
