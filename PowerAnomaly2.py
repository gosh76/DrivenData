import numpy as np
import pandas as pd
import csv
trn = pd.read_csv('F:/PowerAnomaly/train.csv')
sub1 = pd.read_csv('F:/PowerAnomaly/submit.csv')
trn = trn.dropna(axis=0,how='any')
list1 = list(sub1.loc[:,'meter_id'].unique())
listf = []
for x in list1:
    s1 = sub1.loc[sub1.meter_id==x,:]
    list2 = list(s1.loc[:,'Timestamp'].unique())
    print(x)
    t1 = trn.loc[trn.meter_id==x,:]
    print('t1 done')
    s1['Values']=list(t1.loc[t1.Timestamp.isin(list2),'Values'])
    listf.append(s1)
    print('Values done')

tst = pd.concat(listf)
bldg = pd.read_csv('F:/PowerAnomaly/metadata.csv')
bldg.isnull().sum()
bldg.loc[bldg.surface.isnull(),'surface'] = bldg.surface.median()
bldg.loc[bldg.units.isnull(),'units'] = 'VAR'
weat = pd.read_csv('F:/PowerAnomaly/weather.csv')
hols = pd.read_csv('F:/PowerAnomaly/holidays.csv')
hols = hols.drop_duplicates(subset=['Date'],keep='first', inplace=False)
tst = tst.drop(['is_abnormal'],axis=1)


c2 = pd.merge(trn,bldg,on='meter_id',how='inner')
c2 = c2.drop(['Unnamed: 0'],axis=1)
del trn
weat = weat.drop(['Unnamed: 0'],axis=1)
c3 = pd.merge(c2,weat,how='inner',on=['site_id','Timestamp'])
del c2
c3['Date'] = [str(c3.Timestamp.values[i])[:10] for i in range(len(c3))]
hols=hols.drop(['row_id'],axis=1)
c4 = pd.merge(c3,hols,how='left',on='Date')
c4 = c4.drop(['site_id_x','site_id_y'],axis=1)
c4['y'] = [str(c4.Timestamp.values[i])[:4] for i in range(len(c4))]
c4['m'] = [str(c4.Timestamp.values[i])[5:7] for i in range(len(c4))]
c4['d'] = [str(c4.Timestamp.values[i])[8:10] for i in range(len(c4))]
c4['h'] = [str(c4.Timestamp.values[i])[11:13] for i in range(len(c4))]
c4['m1'] = [str(c4.Timestamp.values[i])[14:16] for i in range(len(c4))]
c4 = c4.drop(['Timestamp','Date'],axis=1)
c4.loc[c4.Holiday.isnull(),'Holiday'] = 'None'


weat = weat.drop_duplicates(subset=['site_id','Timestamp'],keep='first', inplace=False)
d2 = pd.merge(tst,bldg,on='meter_id',how='inner')
d2 = d2.drop(['obs_id'],axis=1)
d3 = pd.merge(d2,weat,how='left',on=['site_id','Timestamp'])
d3['Date'] = [str(d3.Timestamp.values[i])[:10] for i in range(len(d3))]
d4 = pd.merge(d3,hols,how='left',on='Date')
d4 = d4.drop(['site_id_x','site_id_y'],axis=1)
d4['y'] = [str(d4.Timestamp.values[i])[:4] for i in range(len(d4))]
d4['m'] = [str(d4.Timestamp.values[i])[5:7] for i in range(len(d4))]
d4['d'] = [str(d4.Timestamp.values[i])[8:10] for i in range(len(d4))]
d4['h'] = [str(d4.Timestamp.values[i])[11:13] for i in range(len(d4))]
d4['m1'] = [str(d4.Timestamp.values[i])[14:16] for i in range(len(d4))]
d4 = d4.drop(['Timestamp','Date'],axis=1)
d4.loc[d4.Temperature.isnull(),'Temperature'] = d4.Temperature.median()
d4.loc[d4.Distance.isnull(),'Distance'] = d4.Distance.median()
d4.loc[d4.Holiday.isnull(),'Holiday'] = 'None'


c4 = c4.drop(['meter_id'],axis=1)
d4 = d4.drop(['meter_id'],axis=1)

"""
import matplotlib.pyplot as plt
c4.boxplot(column='Values')
plt.show()

"""

c4.describe()
c4.loc[c4.Values>80748.0,:].shape
c4['is_abnormal'] = 'False'
c4.loc[c4.Values>80748.0,'is_abnormal'] = 'True'
target = c4.is_abnormal
c4 = c4.drop(['is_abnormal'],axis=1)
c4['meter_description'] = c4.meter_description.astype('category').cat.codes
c4['units'] = c4.units.astype('category').cat.codes
c4['activity'] = c4.activity.astype('category').cat.codes
c4['Holiday'] = c4.Holiday.astype('category').cat.codes

target = target.map({'False':0,'True':1})
d4['meter_description'] = d4.meter_description.astype('category').cat.codes
d4['units'] = d4.units.astype('category').cat.codes
d4['activity'] = d4.activity.astype('category').cat.codes
d4['Holiday'] = d4.Holiday.astype('category').cat.codes


tra = np.array(c4,order='C',copy=False)
tar = np.array(target,order='C',copy=False)
tes = np.array(d4,order='C',copy=False)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=150,verbose=2,random_state=1029)
model = model.fit(tra,tar)

pred = model.predict(tes)
pred = pd.Series(pred)
pred1 = pred.map({0:'False',1:'True'})
sub1['is_abnormal'] = pred1
sub1.to_csv('Power2.csv',index=False)
