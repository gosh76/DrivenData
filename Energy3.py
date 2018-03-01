import numpy as np
import pandas as pd
import csv
trn = pd.read_csv('F:/EnergyConsume/train.csv')
trn.isnull().sum()
trn = trn.dropna(axis=0,how='any')
weat = pd.read_csv('F:/EnergyConsume/weather.csv')
tst = pd.read_csv('F:/EnergyConsume/sub.csv')
meta = pd.read_csv('F:/EnergyConsume/metadata.csv')
hol = pd.read_csv('F:/EnergyConsume/holidays.csv')

#train
trn = trn.drop(['obs_id'],axis=1)
trn1 = pd.merge(trn,meta,on='SiteId',how='inner')
del trn
trn2 = pd.merge(trn1,weat,on=['SiteId','Timestamp'],how='inner')
del trn1
trn2['Date'] = [str(trn2.Timestamp.values[i])[0:10] for i in range(len(trn2))]
hol = hol.drop_duplicates(subset=['SiteId','Date'],keep='first', inplace=False)
trn3 = pd.merge(trn2,hol,on=['SiteId','Date'],how='left')
del trn2
trn3 = trn3.drop(['Unnamed: 0_x','Unnamed: 0_y'],axis=1)
trn3.isnull().sum()
trn3.loc[trn3.Holiday.isnull(),'Holiday'] = 'None'
list1 = ['MondayIsDayOff','TuesdayIsDayOff','WednesdayIsDayOff','ThursdayIsDayOff','FridayIsDayOff','SaturdayIsDayOff','SundayIsDayOff','Holiday']
for x in list1:
    trn3[x] = trn3[x].astype('category').cat.codes

trn3['y'] = [str(trn3.Timestamp.values[i])[:4] for i in range(len(trn3))]
trn3['m'] = [str(trn3.Timestamp.values[i])[5:7] for i in range(len(trn3))]
trn3['d'] = [str(trn3.Timestamp.values[i])[8:10] for i in range(len(trn3))]
trn3['h'] = [str(trn3.Timestamp.values[i])[11:13] for i in range(len(trn3))]
trn3['m1'] = [str(trn3.Timestamp.values[i])[14:16] for i in range(len(trn3))]
trn3 = trn3.drop(['SiteId','Timestamp','Date'],axis=1)
target = trn3.Value
trn3 = trn3.drop(['Value'],axis=1)

target.describe()
target[target>234004.8505].shape
target[target>234004.8505]=234004



#test
tst.isnull().sum()
tst1 = tst.drop(['obs_id','ForecastId','Value'],axis=1)
tst2 = pd.merge(tst1,meta,on='SiteId',how='inner')
del tst1
weat1 = weat.drop_duplicates(subset=['SiteId','Timestamp'],keep='first', inplace=False)
tst3 = pd.merge(tst2,weat1,on=['SiteId','Timestamp'],how='left')
tst3['Date'] = [str(tst3.Timestamp.values[i])[0:10] for i in range(len(tst3))]
tst4 = pd.merge(tst3,hol,on=['SiteId','Date'],how='left')
del tst2,tst3,weat1
tst4.isnull().sum()
tst4 = tst4.drop(['Unnamed: 0_x','Unnamed: 0_y','SiteId','Date'],axis=1)
tst4.loc[tst4.Temperature.isnull(),'Temperature'] = tst4.Temperature.median()
tst4.loc[tst4.Distance.isnull(),'Distance'] = tst4.Distance.median()
tst4.loc[tst4.Holiday.isnull(),'Holiday'] = 'None'
for x in list1:
    tst4[x] = tst4[x].astype('category').cat.codes

tst4['y'] = [str(tst4.Timestamp.values[i])[:4] for i in range(len(tst4))]
tst4['m'] = [str(tst4.Timestamp.values[i])[5:7] for i in range(len(tst4))]
tst4['d'] = [str(tst4.Timestamp.values[i])[8:10] for i in range(len(tst4))]
tst4['h'] = [str(tst4.Timestamp.values[i])[11:13] for i in range(len(tst4))]
tst4['m1'] = [str(tst4.Timestamp.values[i])[14:16] for i in range(len(tst4))]
tst4 = tst4.drop(['Timestamp'],axis=1)

#model
tra = np.array(trn3,order='C',copy=False)
tar = np.array(target,order='C',copy=False)
tes = np.array(tst4,order='C',copy=False)

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100,verbose=2,random_state=79)
model = model.fit(tra,tar)
pred = model.predict(tes)
tst['Value'] = pred
tst.to_csv('F:/EnergyConsume/Energy3.csv',index=False)


