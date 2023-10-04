from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve,auc
from sklearn.model_selection import train_test_split,RandomizedSearchCV,KFold
import sklearn.preprocessing as skl_pre
import sklearn.ensemble as skl_e
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import seaborn as sns


np.random.seed(1)

data = pd.read_csv('train.csv')

X = data.drop(columns=['Lead','Total words'])
y = data['Lead']

def roc(X,y,name):
    X_train,X_val,y_train,y_val =train_test_split(X,y,test_size=0.25)
    model=skl_da.QuadraticDiscriminantAnalysis()
    model.fit(X_train,y_train)
    probs = model.predict_proba(X_val)[:,1]
    #probs = model.predict(X_val)
    fpr, tpr, threshold = roc_curve(y_val, probs,pos_label='Female')
    roc_auc = round(auc(tpr, fpr),2)
    name += ' '+ str(roc_auc)
    plt.plot(tpr,fpr,label=name)

original=roc(X,y,'All')
woGross=roc(X.drop(columns=['Gross']),y,'-Gross')
woNwm=roc(X.drop(columns=['Number words male']),y,'-Number words male')
woNvf=roc(X.drop(columns=['Number words female']),y,'-Number words female')
woYear=roc(X.drop(columns=['Year']),y,'-Year')

Gross=roc(np.array(X['Gross']).reshape(-1,1),y, 'Gross')
Nwm=roc(np.array(X['Number words male']).reshape(-1,1),y,'Number words male')
Nvf=roc(np.array(X['Number words female']).reshape(-1,1),y, 'Number words female')
Year=roc(np.array(X['Year']).reshape(-1,1),y, 'Year')
plt.legend(loc = 'lower right')
plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


"""
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
"""