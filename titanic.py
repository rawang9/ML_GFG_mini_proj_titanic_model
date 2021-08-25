import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
data=pd.read_csv("titanic data.csv")
print(data.head())
print(data.info())
#checking for null val
print(data.isnull().sum())

#check the most depended factor for survived using heatmap
#Numerical Value Analysis
heatmap=sea.heatmap(data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True)
#plt.show()
#Only Fare feature seems to have a significative correlation with the survival probability.
#sibsp - Number of siblings / spouses aboard the Titanic
print(data["SibSp"].unique())
bargraph_sib=sea.factorplot(x="SibSp",y="Survived",data=data,kind="bar",size=8)
bargraph_sib=bargraph_sib.set_ylabels("survaval probability")
# plt.show()
#It seems that passengers having a lot of siblings/spouses have less chance to survive.
#Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive.

#Age
age_visual=sea.FacetGrid(data,col="Survived",size=7)
age_visual=age_visual.map(sea.distplot,"Age")
age_visual=age_visual.set_ylabels("survaval probability")
#plt.show()
# #we can clearly see the peek for young age
# Age distribution seems to be a tailed distribution, maybe a gaussian distribution.

# We notice that age distributions are not the same in the survived and not survived subpopulations. Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived.

# So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.

# It seems that very young passengers have more chance to survive.
#sex
sex_plot = sea.barplot(x="Sex",y="Survived",data=data)
sex_plot =  sex_plot.set_ylabel("Survived_probability")
print(data[["Sex","Survived"]].groupby("Sex").mean())
pclass= sea.factorplot(x="Pclass",y="Survived",data=data,kind="bar")
pclass=pclass.set_ylabels("Survived_prediction")
#Pclass vs Survived by Sex
pss=sea.factorplot(x="Pclass",y="Survived",data=data,hue="Sex",size=8,kind="bar")
pss=pss.set_ylabels("survive prediction")
#Embarked
print(data["Embarked"].isnull().sum())
print(data["Embarked"].value_counts())
print(data["Embarked"].value_counts())
#Fill Embarked with 'S' i.e. the most frequent values
data["Embarked"] = data["Embarked"].fillna("S")
print(data["Embarked"].isnull().sum())
emba=sea.factorplot(x="Embarked",y="Survived",data=data,size=8,kind="bar")
emba=emba.set_ylabels("pred_surv")
# Explore Pclass vs Embarked
g=sea.factorplot("Pclass",col="Embarked",data=data,kind="count",size=16)
g.despine(left=True)
g=g.set_ylabels("survived prediction")
plt.show()
data=pd.read_csv("titanic data.csv")
print(data.head())
print(data.info())
#filling the age colum
mean=data.Age.mean()
std=data.Age.std()
is_null=data.Age.isnull().sum()
rand_age=np.random.randint(mean-std,mean+std,size=is_null)
# fill NaN values in Age column with random values generated
age_slice=data.Age.copy()
age_slice[np.isnan(age_slice)] = rand_age
data.Age=age_slice
print(data.Age.isnull().sum())
#with embarked
data["Embarked"]=data["Embarked"].fillna("S")
col_to_drop=["PassengerId","Name","Ticket","Cabin"]
data.drop(col_to_drop,axis=1,inplace=True)
gender={"male":0,"female":1}
data["Sex"]=data["Sex"].map(gender)
embark={"S":0,"C":1,"Q":2}
data["Embarked"]=data["Embarked"].map(embark)
print(data.head(10))
x=data.drop(data.columns[[0]],axis=1)
y=data.Survived
print(x.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.30,random_state=0)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
#importing the training model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#declaration
lr=LogisticRegression()
dtc=DecisionTreeClassifier()
rfc=RandomForestClassifier(n_estimators=1000,criterion="entropy",random_state=0)
svc=SVC()
knc=KNeighborsClassifier(n_neighbors=5)
#trinag
lr.fit(xtrain,ytrain)
dtc.fit(xtrain,ytrain)
rfc.fit(xtrain,ytrain)
svc.fit(xtrain,ytrain)
knc.fit(xtrain,ytrain)
#predicting
lr_pred = lr.predict(xtest)
dtc_pred = dtc.predict(xtest)
rfc_pred = rfc.predict(xtest)
svc_pred = svc.predict(xtest)
knc_pred = knc.predict(xtest)
from sklearn.metrics import accuracy_score
# finding accuracy
from sklearn.metrics import accuracy_score

logreg_acc = accuracy_score(ytest, lr_pred)
svc_classifier_acc = accuracy_score(ytest, svc_pred)
dt_classifier_acc = accuracy_score(ytest, dtc_pred)
knn_classifier_acc = accuracy_score(ytest, knc_pred)
rf_classifier_acc = accuracy_score(ytest, rfc_pred)
print ("Logistic Regression : ", round(logreg_acc*100, 2))
print ("Support Vector      : ", round(svc_classifier_acc*100, 2))
print ("Decision Tree       : ", round(dt_classifier_acc*100, 2))
print ("K-NN Classifier     : ", round(knn_classifier_acc*100, 2))
print ("Random Forest       : ", round(rf_classifier_acc*100, 2))