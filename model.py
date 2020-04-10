# Importing the libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('soildata.csv')

le = LabelEncoder()
dataset['Treatment'] = le.fit_transform(dataset['Treatment'])

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values

missingvalues = SimpleImputer(
    missing_values=np.nan, strategy='mean', verbose=0)
missingvalues = missingvalues.fit(X[:, :])
X[:, :] = missingvalues.transform(X[:, :])

#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Decision Tree Classification to the Training set
classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_DT.fit(X_train, y_train)
y_pred = classifier_DT.predict(X_test)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
asDT = 100*accuracy_score(y_pred, y_test)

# Saving model to disk
pickle.dump(classifier_DT, open('model_DT.pkl', 'wb'))

# Fitting K-NN to the Training set
classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_KNN.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_KNN.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
asKNN = 100*accuracy_score(y_pred, y_test)

# Saving model to disk
pickle.dump(classifier_KNN, open('model_KNN.pkl', 'wb'))

# Fitting SVM to the Training set
classifier_SVM = SVC(kernel='linear', random_state=0)
classifier_SVM.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_SVM.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
asSVM = 100*accuracy_score(y_pred, y_test)
# Saving model to disk
pickle.dump(classifier_SVM, open('model_SVM.pkl', 'wb'))


# Fitting Naive Bayes to the Training set
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_NB.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
asNB = 100*accuracy_score(y_pred, y_test)

# Saving model to disk
pickle.dump(classifier_NB, open('model_NB.pkl', 'wb'))

accuracy = {"Decision Tree": asDT, "SVM": asSVM,
            "KNN": asKNN, "Naive Bayes": asNB}

