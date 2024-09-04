import numpy as np
import pandas as pd

dataset = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")
dataset.pop('PassengerId')
dataset.pop('Name')
dataset.pop('Ticket')
dataset.pop('Embarked')
dataset.pop('Cabin')

dataset_test.pop('PassengerId')
dataset_test.pop('Name')
dataset_test.pop('Ticket')
dataset_test.pop('Embarked')
dataset_test.pop('Cabin')

check = pd.read_csv("gender_submission.csv")
check.pop('PassengerId')
check_y = check.iloc[:, :].values

Test = dataset_test.iloc[:, :].values

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# adding missing data
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
column_to_impute = X[:, 2].reshape(-1, 1)
simple_imputer.fit(column_to_impute)
X[:, 2] = simple_imputer.transform(column_to_impute).ravel()
simple_imputer.fit(Test[:, [2, -1]])
Test[:, [2, -1]] = simple_imputer.transform(Test[:, [2, -1]])

# creating dummy values of sex
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
Test = np.array(ct.fit_transform(Test))

# splitting dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Test = sc.transform(Test)

# training the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# predicting test set results (train.csv)
y_pred = classifier.predict(X_test)

# predicting test set results (test.csv)
y_pred_test = classifier.predict(Test)

from sklearn.metrics import confusion_matrix, accuracy_score
# accuracy of the test set (train.csv)
print("train.csv")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

print()

# accuracy of the test set (test.csv)
print("test.csv")
cm2 = confusion_matrix(check_y, y_pred_test)
print(cm2)
print(accuracy_score(check_y, y_pred_test))

print()

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

with open('submission.csv', 'w') as file:
    file.write('PassengerId,Survived\n')
    for i in range (0, len(y_pred_test)):
        file.write(str(892+i))
        file.write(',')
        file.write(str(y_pred_test[i]))
        file.write('\n')

# feature scaling worsen the accuracy score
# adding 'embarked' column doesn't change the score
