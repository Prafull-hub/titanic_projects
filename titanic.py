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
import random
from sklearn.base import BaseEstimator, TransformerMixin

# weights = []
from sklearn import set_config
set_config(transform_output="pandas")


train = pd.read_csv("c:/Users/mishr/Downloads/train.csv")
test = pd.read_csv("c:/Users/mishr/Downloads/test.csv")

x = train.drop(columns='Survived')
y = train['Survived']


x['Title'] = x['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, column="Age", random_state=42, add_indicator=True):
        self.column = column
        self.random_state = random_state
        self.add_indicator = add_indicator

    def fit(self, x, y=None):
        x = x.copy()
        self.mean_ = x[self.column].mean()
        self.std_ = x[self.column].std()
        return self

    def transform(self, x):
        x = x.copy()
        rng = np.random.RandomState(self.random_state)

        if self.add_indicator:
            missing_flag = x[self.column].isnull().astype(int)

        n_missing = x[self.column].isnull().sum()
        if n_missing > 0:
            random_vals = rng.uniform(
            self.mean_ - self.std_,
            self.mean_ + self.std_,
            n_missing)
            x.loc[x[self.column].isnull(), self.column] = random_vals

        if self.add_indicator:
            x[self.column + '_missing'] = missing_flag

        return x

    def get_feature_names_out(self, input_features=None):
        if self.add_indicator:
            return np.array([self.column, self.column + '_missing'])
        return np.array([self.column])

def get_family(x):
    total_family = x["SibSp"] + x["Parch"]
    
    if total_family == 0:
        return 0
    elif total_family < 3:
        return 1
    else:
        return 2

x['Family_Category'] = x.apply(get_family, axis=1)
test['Family_Category'] = test.apply(get_family, axis=1)

x = x.drop(columns=['Cabin','Ticket', "PassengerId","SibSp", "Parch", "Name"])
test = test.drop(columns=['Cabin','Ticket', "PassengerId","SibSp", "Parch", "Name"])

title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Royalty', 'Countess': 'Royalty', 'Sir': 'Royalty',
    'Don': 'Royalty', 'Dona': 'Royalty', 'Jonkheer': 'Royalty',
    'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer', 'Dr': 'Officer', 'Rev': 'Officer'
}
x['Title'] = x['Title'].replace(title_map)
test['Title'] = test['Title'].replace(title_map)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.head())



tnf1 = ColumnTransformer([
    ('filing_age', Imputer(column="Age", add_indicator=True), ['Age']),
    ('filing_embarked', SimpleImputer(strategy="most_frequent"), ['Embarked']),
    ('filing_fare', SimpleImputer(strategy="median"), ['Fare']),
], remainder='passthrough', verbose_feature_names_out=False)

print(x_train.head())
tnf2 = ColumnTransformer([(
    'ohe columns',OneHotEncoder(handle_unknown='ignore', sparse_output=False),['Title', 'Sex', 'Embarked']
)],remainder='passthrough')
# print(x_train.head())

tnf3 = StandardScaler()
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


clf1 = RandomForestClassifier(random_state=7,max_depth=None,n_estimators=100)
clf2 = DecisionTreeClassifier(max_depth=7)
clf3 = LogisticRegression(random_state=42)
clf4 = SVC(random_state=42,C=2,probability=True)


vc = VotingClassifier(estimators=[('rf', clf1), ('lr', clf3), ('svc', clf4)], voting='hard',weights=[1,1,1])
tnf4 = RandomForestClassifier(random_state=42)
pipe =Pipeline([
    ('tnf1',tnf1),
    ('tnf2',tnf2),
    ('tnf3',tnf3),
    ('tnf4',vc),
])
pipe.fit(x_train,y_train)

from sklearn.metrics import accuracy_score

# check on your holdout split
print("holdout acc:", accuracy_score(y_test, pipe.predict(x_test)))
print("cv acc:", cross_val_score(pipe, x, y, cv=5).mean())

# refit on all training data, then predict test
pipe.fit(x, y)
y_pred = pipe.predict(test)

sub = pd.DataFrame({
    "PassengerId": pd.read_csv("c:/Users/mishr/Downloads/test.csv")["PassengerId"],
    "Survived": y_pred
})
sub.to_csv("submission.csv", index=False)
print(sub.shape)  # (418, 2)