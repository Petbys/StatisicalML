import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV,KFold
import sklearn.preprocessing as skl_pre
import sklearn.ensemble as skl_e
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import matplotlib.pyplot as plt
def predict(model, X_val, y_val):
    # Print model parameters.
    print(model)
    # Predict the hard class.
    predict_class = model.predict(X_val)
    # Confusion matrix.
    print("\nConfusion matrix:")
    print(pd.crosstab(y_val, predict_class, margins=True))
    
    # Accuracy.
    print(f"\nAccuracy: {np.mean(predict_class == y_val):.5f}")
    error = np.mean(predict_class != y_val)
    print('Error: %.3f' % error)

# Gather data
np.random.seed(1)

data = pd.read_csv('train.csv', na_values='?', dtype = {'ID':str}).dropna().reset_index(drop=True)
X = data.drop(columns=['Lead'])
y = data['Lead']


X_train,X_val,y_train,y_val =train_test_split(X,y,test_size=0.25)
""" 
# scale data
sc = skl_pre.StandardScaler()
X_train=sc.fit_transform(X_train)
X_val=sc.fit_transform(X_val)

#logistic regressions model, lbgfs wo tuning
model=skl_lm.LogisticRegression(solver='lbfgs',max_iter=1000)
model.fit(X_train,y_train)
prediction=predict(model, X_val, y_val)

#Tune hyperparameters
model.get_params()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

grid = dict(solver=solvers,penalty=penalty,C=c_values)
model_random= RandomizedSearchCV(estimator=model,param_distributions=grid,cv=10)
model_random.fit(X_train, y_train) 
grid_result = model_random.fit(X, y)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
best = grid_result.best_params_
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
best=skl_lm.LogisticRegression(solver= grid_result.best_params_['solver'],penalty=grid_result.best_params_['penalty'],C=grid_result.best_params_['C'])
best.fit(X_train,y_train)
print(f'best:  \n')
predict(best, X_val, y_val)
print(f'original:  \n')
predict(model,X_val,y_val)

# CV Logistic Regression
n_fold = 10
cv = KFold(n_splits = n_fold, random_state = 1, shuffle = True)
missclassifications = np.zeros((n_fold, 1))

for i, (train_index, val_index) in enumerate(cv.split(X)):
   X_train, X_val = X.iloc[train_index], X.iloc[val_index]
   Y_train, Y_val = y.iloc[train_index], y.iloc[val_index]
 
   model = skl_lm.LogisticRegression(solver='lbfgs')
   model.fit(X_train, Y_train)
   prediction = model.predict(X_val)
   missclassifications[i] = np.mean(prediction !=Y_val)
 
error = np.mean(missclassifications)
print(error)
print(100-100*error)

for i, (train_index, val_index) in enumerate(cv.split(X)):
   X_train, X_val = X.iloc[train_index], X.iloc[val_index]
   Y_train, Y_val = y.iloc[train_index], y.iloc[val_index]
 
   model = skl_lm.LogisticRegression()
   model.fit(X_train, Y_train)
   prediction = model.predict(X_val)
   missclassifications[i] = np.mean(prediction !=Y_val)
 
error = np.mean(missclassifications)
print(f'{error} wo tuning')
print(100-100*error)

 """
#CV 

n_fold = 10
models = []

models.append(skl_e.RandomForestClassifier(n_estimators = 500, min_samples_split =  5, min_samples_leaf = 1, max_features = None))
models.append(skl_nb.KNeighborsClassifier(metric = 'minkowski',n_neighbors = 5))
models.append(skl_da.QuadraticDiscriminantAnalysis())
models.append(skl_lm.LogisticRegression(solver='lbfgs', max_iter=10000))


missclassifications = np.zeros((n_fold, len(models)))
cv = KFold(n_splits = n_fold, random_state = 1, shuffle = True)

for i, (train_index, val_index) in enumerate(cv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    for m in range(np.shape(models)[0]):
        model = models[m]

        if m==0:
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
            missclassifications[i, m] = np.mean(prediction != y_val)

        elif m==1:
            model.fit(skl_pre.normalize(X_train), y_train)
            prediction = model.predict(skl_pre.normalize(X_val))
            missclassifications[i, m] = np.mean(prediction != y_val)

        elif m==2:
            model.fit(X_train.drop(columns=['Total words']), y_train)
            prediction = model.predict(X_val.drop(columns=['Total words']))
            missclassifications[i, m] = np.mean(prediction != y_val)

        elif m==3:
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
            missclassifications[i, m] = np.mean(prediction != y_val)

plt.boxplot(missclassifications)
plt.title('Cross validation errors for different methods')
plt.xticks(np.arange(4) +1, ('Random forest', 'k-NN', 'QDA', 'LR'))
plt.ylabel('validation error')
plt.show()
