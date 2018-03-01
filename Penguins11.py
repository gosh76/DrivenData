import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import BaggingRegressor
pd.set_option('display.max_columns', 292)
pd.set_option('display.width', 140)
pd.set_option('display.max_rows', 300)
trainO = pd.read_csv('~/Desktop/sfolder/DrivenData-Penguins/trainobs.csv')
trainE = pd.read_csv('~/Desktop/sfolder/DrivenData-Penguins/traine_n.csv')
trainN = pd.read_csv('~/Desktop/sfolder/DrivenData-Penguins/trainnest.csv')
cols = trainN.columns.tolist()
for a in cols[2:]:
    for x in trainN.loc[trainN[a].isnull(),:].index.tolist():
        if len(trainO.loc[(trainO.site_id == trainN.ix[x].site_id) & (trainO.common_name == trainN.ix[x].common_name) & (trainO.season_starting == int(a)) & \
        (trainO.count_type == 'chicks'),'penguin_count'].values) > 0:
            trainN.loc[x,a] = trainO.loc[(trainO.site_id == trainN.ix[x].site_id) & (trainO.common_name == trainN.ix[x].common_name) & \
            (trainO.season_starting == int(a)),'penguin_count'].values[0]/float(1.5)
        elif len(trainO.loc[(trainO.site_id == trainN.ix[x].site_id) & (trainO.common_name == trainN.ix[x].common_name) & (trainO.season_starting == int(a))\
         & (trainO.count_type == 'adults'),'penguin_count'].values) > 0:
            trainN.loc[x,a] = trainO.loc[(trainO.site_id == trainN.ix[x].site_id) & (trainO.common_name == trainN.ix[x].common_name) & \
            (trainO.season_starting == int(a)),'penguin_count'].values[0]/float(3.5)

sites = trainN.site_id
species = trainN.common_name
trainN = trainN.drop(['site_id','common_name'],axis = 1)
trainN1 = trainN.fillna(method='ffill',axis=1)
trainN2 = trainN1.fillna(method='bfill',axis=1)
trainN2.loc[trainN2['1895'].isnull(),'1895'] = trainN2.loc[:,'1895'].median()
trainN2.loc[trainN2['1983'].isnull(),'1983'] = trainN2.loc[:,'1983'].median()
trainN2.loc[trainN2['2010'].isnull(),'2010'] = trainN2.loc[:,'2010'].median()
trainN5 = trainN2.fillna(method='ffill',axis=1)
trainN6 = trainN5.fillna(method='bfill',axis=1)
trarr1 = np.array(trainN6)
tra1 = trarr1[:,0:54]
tar1 = trarr1[:,54]
modelb = BaggingRegressor(base_estimator=None,n_estimators=100,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,oob_score=False,\
warm_start=False,n_jobs=-1,random_state=29,verbose=2)
modelb = modelb.fit(tra1,tar1)
pred1 = modelb.predict(trarr1[:,1:])
trainN6['2014'] = pred1
trarr2 = np.array(trainN6)
tra2 = trarr2[:,0:55]
tar2 = trarr2[:,55]
modelb1 = BaggingRegressor(base_estimator=None,n_estimators=100,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,oob_score=False,\
warm_start=False,n_jobs=-1,random_state=29,verbose=2)
modelb1 = modelb1.fit(tra2,tar2)
pred2 = modelb1.predict(trarr2[:,1:])
trainN6['2015'] = pred2
trarr3 = np.array(trainN6)
tra3 = trarr3[:,0:56]
tar3 = trarr3[:,56]
modelb2 = BaggingRegressor(base_estimator=None,n_estimators=100,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,oob_score=False,\
warm_start=False,n_jobs=-1,random_state=29,verbose=2)
modelb2 = modelb2.fit(tra3,tar3)
pred3 = modelb2.predict(trarr3[:,1:])
trainN6['2016'] = pred3
trarr4 = np.array(trainN6)
tra4 = trarr4[:,0:57]
tar4 = trarr4[:,57]
modelb3 = BaggingRegressor(base_estimator=None,n_estimators=100,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,oob_score=False,\
warm_start=False,n_jobs=-1,random_state=29,verbose=2)
modelb3 = modelb3.fit(tra4,tar4)
pred4 = modelb3.predict(trarr4[:,1:])
trainN6['2017'] = pred4

submission = pd.DataFrame(sites,columns=['site_id'])
submission['common_name'] = species
submission['2014'] = trainN6['2014']
submission['2015'] = trainN6['2015']
submission['2016'] = trainN6['2016']
submission['2017'] = trainN6['2017']
submission.to_csv('Penguins11.csv',index=False)

