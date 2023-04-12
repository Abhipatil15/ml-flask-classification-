import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('loan_data.csv')
df.isnull().sum()
df['credit_policy'].value_counts()
df['purpose'].value_counts()
df["int_rate"].value_counts()
df['installment'].value_counts()
df['dti'].value_counts()
df['fico'].value_counts()
df['days_with_cr_line'].value_counts()
df['revol_bal'].value_counts()
df['revol_util'].value_counts()
df['inq_last_6mths'].value_counts()
df['delinq_2yrs'].value_counts()
df['pub_rec'].value_counts()

df.shape
Numcol=[]
for i in df.dtypes.index:
    if df.dtypes[i]!='object':
        Numcol.append(i)
Numcol 
catcol=[]
for i in df.dtypes.index:
    if df.dtypes[i]=='object':
        catcol.append(i)
catcol

f=df[['credit_policy','int_rate','installment','log_annual_inc','dti','fico','days_with_cr_line','revol_bal','revol_util','inq_last_6mths','delinq_2yrs','pub_rec','not_fully_paid']]
from scipy.stats import zscore
z=abs(zscore(f))
z
newdf=df[(z<3).all(axis=1)]
newdf
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
newdf[catcol]=oe.fit_transform(newdf[catcol])
newdf.head()

newdf.skew()

s=['credit_policy', 'int_rate', 'installment', 'log_annual_inc', 'dti', 'fico','days_with_cr_line', 'revol_bal', 'revol_util', 'inq_last_6mths','pub_rec', 'delinq_2yrs']
from sklearn.preprocessing import PowerTransformer
scaler=PowerTransformer(method='yeo-johnson')
newdf[s]=scaler.fit_transform(newdf[s].values)
x = newdf.drop('not_fully_paid',axis=1)
x
y = newdf['not_fully_paid']
y
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    
    train=model.score(xtrain,ytrain)
    test=model.score(xtest,ytest)
    
    print(f"Traning accuracy:{train}\n Testing accuracy:{test}\n\n")
    print(confusion_matrix(ytest,ypred))
    print(classification_report(ytest,ypred))
    print(f"Accuracy:{accuracy_score(ytest,ypred)}")
    return model

mymodel(RandomForestClassifier(criterion="entropy",max_depth=1,min_samples_leaf=9))

def mymodel(model):
    model.fit(xtrain,ytrain)
    return model


def makepredict():
    RFC=RandomForestClassifier()
    model=mymodel(RFC)
    return model