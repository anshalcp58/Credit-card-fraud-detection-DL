
import pandas as pd
from sklearn.model_selection import train_test_split

mydata = pd.read_csv("C:\\Users\\lenshif\\Desktop\\TRANSACTIONX\\transactionx\\creditcard.csv")


df = pd.DataFrame(mydata)

X = mydata.drop("Class", axis=1)
y = mydata["Class"]


X= X[:25000]
y= y[:25000]
print("-------------------%----------")
print(y)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.10, random_state=42)
print("split")
from sklearn.neural_network import MLPClassifier
print("import")
regressor = MLPClassifier()
print("regr")
regressor.fit(X_train, y_train)

print("fit")
y_pred = regressor.predict(X_test)
print(y_pred,"==========================", "value")
