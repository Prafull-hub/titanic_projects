import pandas as pd 
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
train = pd.read_csv("c:/Users/mishr/Downloads/train.csv")
test = pd.read_csv("c:/Users/mishr/Downloads/test.csv")

x = train.drop(columns='Survived')
y = train['Survived']

x = x.drop(columns=['Cabin', 'Name','PassengerId','Ticket'])



print(x.head())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)



tnf1 = ColumnTransformer([(
    'filing age',SimpleImputer(),[2]
)], remainder='passthrough')

tnf2 = ColumnTransformer([(
    'ohe columns',OneHotEncoder(handle_unknown='ignore', sparse_output=False),[2,6]
)],remainder='passthrough')
print(x_train.head())

tnf3 = StandardScaler()

tnf4 = RandomForestClassifier(random_state=42)

pipe =Pipeline([
    ('tnf1',tnf1),
    ('tnf2',tnf2),
    ('tnf3',tnf3),
    ('tnf4',tnf4),
])
pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
from sklearn.metrics import accuracy_score

# Get Precision, Recall, and F1-Score for every class at once
print(accuracy_score(y_test, y_pred))
# print(x_train.head())
cv_scores = cross_val_score(pipe, x_train, y_train, cv=5, scoring='accuracy')

print("--- Cross-Validation Results ---")
print(f"Scores for each fold: {cv_scores}")
print(f"Mean CV Accuracy:     {cv_scores.mean():.4f}")
print(f"Standard Deviation:   {cv_scores.std():.4f}\n")
