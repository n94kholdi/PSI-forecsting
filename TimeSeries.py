import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


Data = pd.read_csv(r'./psi_df_2016_2019.csv')

def forcasting_features(n_days, psi):

    new_features = []
    new_labels = []
    for i in range(n_days, len(psi)):
        new_features.append(psi[i - n_days:i])
        new_labels.append(psi[i])

    return (new_features, new_labels)

Start_day = int(input('Please enter your start day between 2000 and 30000:'
                    '\n(It would predict 500 days longer)\n').strip())

location_feature = 'central'

# print(Start_day)
# exit(0)

###########################################################
######################## SVR ##############################

All_MSE = []
Best_test = []
Best_pred = []
Best_score = -2000000000

print('ML Method : SVR\n')
for day in range(5, 25):
    #########################################################
    ############### Feature Extracting ######################
    n_days = day
    (new_features, new_labels) = forcasting_features(n_days, list(Data[location_feature]))#, list(Data['Volume']))

    #########################################################
    ############### Split to Train and Test Data ############

    ### We use 2000 days before Start-day for prediction
    X_train = new_features[Start_day - 2000:Start_day]
    Y_train = new_labels[Start_day - 2000:Start_day]

    ### We predict 500 days after Start_day
    X_test = new_features[Start_day:Start_day + 500]
    Y_test = np.array(new_labels[Start_day:Start_day + 500])

    #########################################################
    ############## Forcasting Times Series ##################

    reg = make_pipeline(StandardScaler(), SVR(kernel='linear', degree=2, C=1.0, epsilon=0.2))
    reg.fit(X_train, Y_train)

    pred = reg.predict(X_test)
    MSE = np.sqrt(np.sum((pred - Y_test)**2))
    cross_val = np.mean(cross_val_score(reg, X_train, Y_train, cv=5))
    All_MSE.append(cross_val)

    if Best_score < cross_val:
        Best_score = cross_val
        Best_pred = pred
        Best_test = Y_test
    # print(n_days, ':', reg.score(X_test, Y_test))
    # break

with open('SVR-result.csv', 'w') as csvfile:
    fieldnames = ['predict-value', 'PSI-value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer = csv.writer(csvfile)
    writer.writerows(zip(pred, Y_test))

plt.plot(Best_pred, label='pred')
plt.plot(Best_test, label='test')
plt.title('SVR')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()
print(Best_score)
# print(reg.predict(X_test[0:4]), Y_test[0:4])

# exit(0)
###########################################################
######################## KNN ##############################

All_MSE = []
Best_test = []
Best_pred = []
Best_score = -2000000000

print('ML Method : knn regression\n')

for day in range(5,25):
    #########################################################
    ############### Feature Extracting ######################
    n_days = day
    (new_features, new_labels) = forcasting_features(n_days, list(Data[location_feature]))

    #########################################################
    ############### Split to Train and Test Data ############

    ### We use 2000 days before Start-day for prediction
    X_train = new_features[Start_day - 2000:Start_day]
    Y_train = new_labels[Start_day - 2000:Start_day]

    ### We predict 500 days after Start_day
    X_test = new_features[Start_day:Start_day + 500]
    Y_test = np.array(new_labels[Start_day:Start_day + 500])

    #########################################################
    ############## Forcasting Times Series ##################
    knn_reg = KNeighborsRegressor(n_neighbors=15)
    knn_reg.fit(X_train, Y_train)

    pred = knn_reg.predict(X_test)
    MSE = np.sqrt(np.sum((pred - Y_test) ** 2))
    cross_val = np.mean(cross_val_score(knn_reg, X_train, Y_train, cv=5))
    # print(n_days, ':', MSE)
    All_MSE.append(cross_val)

    if Best_score < cross_val:
        Best_score = cross_val
        Best_pred = pred
        Best_test = Y_test

    # print(n_days, ':', reg.score(X_test, Y_test))
    # break

with open('KNN-reg_result.csv', 'w') as csvfile:
    fieldnames = ['predict-value', 'PSI-value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer = csv.writer(csvfile)

    writer.writerows(zip(pred, Y_test))

# plt.plot(All_MSE)
plt.plot(Best_pred, label='pred')
plt.plot(Best_test, label='test')
plt.title('KNN-reg')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()
print(Best_score)

# print(knn_reg.predict(X_test[0:4]), Y_test[0:4])

