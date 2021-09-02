import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("data/airline.csv")
data = data.drop(['Flight', 'Time'], axis=1)
data= data[:10000]


X = data.loc[:,['Airline','AirportFrom','AirportTo','DayOfWeek','Length',]]
y = data.Delay

column_trans = make_column_transformer(
    (OneHotEncoder(sparse=False,handle_unknown='ignore'), ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']),
    remainder='passthrough')

logreg = LogisticRegression(solver='lbfgs',max_iter=1000)
randomForestclf = RandomForestClassifier(max_depth=2, random_state=0)
pipe = make_pipeline(column_trans, logreg)

X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=42)

print(cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean())

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

print("finish")


