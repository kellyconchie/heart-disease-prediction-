import pandas as pd
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("heart.csv")

# I am going to use the following bit of code to check for missing data
# 0 means no missing data
print(df.isna().sum())

# Removing the rows that contain missing data
df = df.dropna(axis=0)

# Checking the shape of the dataframe after the removal of data
print(df.shape)

# Converting character data to numerical
df['ChestPain'] = df['ChestPain'].replace(['typical'], 1)
df['ChestPain'] = df['ChestPain'].replace(['nontypical'], 2)
df['ChestPain'] = df['ChestPain'].replace(['asymptomatic'], 3)
df['ChestPain'] = df['ChestPain'].replace(['nonanginal'], 4)

df['Thal'] = df['Thal'].replace(['normal'], 1)
df['Thal'] = df['Thal'].replace(['fixed'], 2)
df['Thal'] = df['Thal'].replace(['reversable'], 3)

df['AHD'] = df['AHD'].replace(['Yes'], 1)
df['AHD'] = df['AHD'].replace(['No'], 0)

# Separating the data into columns to be evaluated and the result result column is the Y column
X = df.drop('AHD', axis=1)
y = df['AHD']

# Preparing data for analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("")


# Stacking Classifier
RANDOM_SEED = 42

clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = DecisionTreeClassifier()
clf3 = RandomForestClassifier(random_state=RANDOM_SEED)
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], use_probas=True,
                            meta_classifier=lr, random_state=RANDOM_SEED)

print('5-fold Cross Validation and Accuracy Scores:\n')

for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN', 'Decision Tree', 'Random Forest', 'Stacking Classifier']):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Cross Validation scores: ", scores, label)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))


# Precision Report for KNN
print("Classification Report for KNN: ")
clf1.fit(X_train, y_train)
clf1_predicted = clf1.predict(X_test)
clf1_conf_matrix = confusion_matrix(y_test, clf1_predicted)
clf1_acc_score = accuracy_score(y_test, clf1_predicted)
print(classification_report(y_test, clf1_predicted))

# Precision Report for Decision Tree
print("Classification Report for Decision Tree: ")
clf2.fit(X_train, y_train)
clf2_predicted = clf2.predict(X_test)
clf2_conf_matrix = confusion_matrix(y_test, clf2_predicted)
clf2_acc_score = accuracy_score(y_test, clf2_predicted)
print(classification_report(y_test, clf2_predicted))

# Precision Report for Random Forest
print("Classification Report for Random Forest: ")
clf3.fit(X_train, y_train)
clf3_predicted = clf3.predict(X_test)
clf3_conf_matrix = confusion_matrix(y_test, clf3_predicted)
clf3_acc_score = accuracy_score(y_test, clf3_predicted)
print(classification_report(y_test, clf3_predicted))

# Precision Report for Stacking
print("Classification Report for Stacking: ")
sclf.fit(X_train, y_train)
sclf_predicted = sclf.predict(X_test)
sclf_conf_matrix = confusion_matrix(y_test, sclf_predicted)
scv_acc_score = accuracy_score(y_test, sclf_predicted)
print(classification_report(y_test, sclf_predicted))
