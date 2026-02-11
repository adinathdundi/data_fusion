##Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier 
 
##Load Dataset 
df = pd.read_csv('data/winequality.csv') 
features = df.columns[:-1]

label = df.columns[-1]
##separate features and label
X = df[features]
Y = df[label]

##Split into train(90%) and test(10%) sets 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


##Build three classifiers; Decision tree, SVM and Logistic regression. 
DT = DecisionTreeClassifier()
SVM = svm.SVC()
LR = LogisticRegression()

##Check performance of classifiers and if necessary perform hyperparameter tuning(Check sklearn documentations for hyperparameter descriptions)

DT.fit(x_train,y_train)
pred_dt = DT.predict(x_test)
accuracy_score(y_test,pred_dt)

SVM.fit(x_train,y_train)
pred_dt = SVM.predict(x_test)
accuracy_score(y_test,pred_dt)

LR.fit(x_train,y_train)
pred_dt = LR.predict(x_test)
accuracy_score(y_test,pred_dt)

##Apply Decision Fusion

#Majority Voting 
# --Hard Voting: Predict the class with the largest sum of votes from models 
# --Soft Voting: Predict the class with the largest summed probability from models 
fusion_1 = VotingClassifier(estimators=[('dt', DT), ('svm', SVM), ('lr', LR)], voting='hard')
fusion_1.fit(x_train,y_train)
fusion_1.predict(x_test) #Check sklearn documentation for more information

#Weighted Voting

#Bayesian Consensus

#Dempster-Shafer’s Theory of evidence


##Evaluate and Report performance of classifiers before and after applying fusion
