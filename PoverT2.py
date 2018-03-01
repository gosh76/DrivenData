import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingClassifier

#ModelA
#train & model
trnhholdA = pd.read_csv("~/PoverT/A_hhold_train.csv")
trnindivA = pd.read_csv("~/PoverT/A_indiv_train.csv")
trnindivA = trnindivA.drop(['iid','poor','country'],axis=1)
c1 = pd.merge(trnhholdA,trnindivA,on='id',how='left')
c1.isnull().sum()
n1 = c1.isnull().sum()
n1[n1>0]#'OdXpbPGJ'
c1.dtypes.loc['OdXpbPGJ']#dtype('float64')
c1.loc[c1.OdXpbPGJ.isnull(),'OdXpbPGJ'] = c1.OdXpbPGJ.median()
m1 = c1.dtypes
l1 = list(m1[m1==object].index)
for x in l1:
    c1[x] = c1[x].astype('category').cat.codes

target = c1.poor
c1 = c1.drop(['id','poor','country'],axis=1)
traA = np.array(c1,order='C',copy=False)
tarA = np.array(target,order='C',copy=False)
modelA = GradientBoostingClassifier(n_estimators=150,verbose=2)
modelA = modelA.fit(traA,tarA)
#test & prediction
tsthholdA = pd.read_csv("~/PoverT/A_hhold_test.csv")
tstindivA = pd.read_csv("~/PoverT/A_indiv_test.csv")
tstindivA = tstindivA.drop(['iid','country'],axis=1)
d1 = pd.merge(tsthholdA,tstindivA,on='id',how='left')
t1 = d1.isnull().sum()
t1[t1>0]#'OdXpbPGJ'
d1.loc[d1.OdXpbPGJ.isnull(),'OdXpbPGJ'] = d1.OdXpbPGJ.median()
idsA = d1.id
d1 = d1.drop(['id','country'],axis=1)
s1 = d1.dtypes
u1 = list(s1[s1==object].index)
for y in u1:
    d1[y] = d1[y].astype('category').cat.codes

tesA = np.array(d1,order='C',copy=False)
predA = modelA.predict_proba(tesA)
#submitA
subA = pd.DataFrame(idsA,columns=['id'])
subA['poor'] = predA[:,1]

ids1 = tsthholdA.id
listA = []
for i in ids1:
    listA.append(subA.loc[subA.id==i,'poor'].mean())

subAf = pd.DataFrame(ids1,columns=['id'])
subAf['country'] = 'A'
subAf['poor'] = listA


#ModelB

#train & model
trnhholdB = pd.read_csv("~/PoverT/B_hhold_train.csv")
trnindivB = pd.read_csv("~/PoverT/B_indiv_train.csv")
trnindivB = trnindivB.drop(['iid','poor','country'],axis=1)
c2 = pd.merge(trnhholdB,trnindivB,on='id',how='left')
c2.isnull().sum()
n2 = c2.isnull().sum()
n2[n2>0]
v1 = n2[n2>0]
c2.dtypes.loc[list(v1.index)]
for z in list(v1.index):
    c2.loc[c2[z].isnull(),z] = c2[z].median()

m2 = c2.dtypes
l2 = list(m2[m2==object].index)
for x in l2:
    c2[x] = c2[x].astype('category').cat.codes

target2 = c2.poor
c2 = c2.drop(['id','poor','country'],axis=1)
traB = np.array(c2,order='C',copy=False)
tarB= np.array(target2,order='C',copy=False)
modelB = GradientBoostingClassifier(n_estimators=150,verbose=2)
modelB = modelB.fit(traB,tarB)

#test & prediction
tsthholdB = pd.read_csv("~/PoverT/B_hhold_test.csv")
tstindivB = pd.read_csv("~/PoverT/B_indiv_test.csv")
tstindivB = tstindivB.drop(['iid','country'],axis=1)
d2 = pd.merge(tsthholdB,tstindivB,on='id',how='left')
t2 = d2.isnull().sum()
t2[t2>0]
e1 = t2[t2>0]
c2.dtypes.loc[list(e1.index)]
for z in list(e1.index):
    d2.loc[d2[z].isnull(),z] = d2[z].median()

idsB = d2.id
d2 = d2.drop(['id','country'],axis=1)
s2 = d2.dtypes
u2 = list(s2[s2==object].index)
for y in u2:
    d2[y] = d2[y].astype('category').cat.codes

tesB = np.array(d2,order='C',copy=False)
predB = modelB.predict_proba(tesB)

#submitB
subB = pd.DataFrame(idsB,columns=['id'])
subB['poor'] = predB[:,1]
ids2 = tsthholdB.id
listB = []
for i in ids2:
    listB.append(subB.loc[subB.id==i,'poor'].mean())

subBf = pd.DataFrame(ids2,columns=['id'])
subBf['country'] = 'B'
subBf['poor'] = listB

#ModelC
#train & model
trnhholdC = pd.read_csv("~/PoverT/C_hhold_train.csv")
trnindivC = pd.read_csv("~/PoverT/C_indiv_train.csv")
trnindivC = trnindivC.drop(['iid','poor','country'],axis=1)
c3 = pd.merge(trnhholdC,trnindivC,on='id',how='left')
c3.isnull().sum()
n3 = c3.isnull().sum()
n3[n3>0]
m3 = c3.dtypes
l3 = list(m3[m3==object].index)
for x in l3:
    c3[x] = c3[x].astype('category').cat.codes

target3 = c3.poor
c3 = c3.drop(['id','poor','country'],axis=1)
traC = np.array(c3,order='C',copy=False)
tarC= np.array(target3,order='C',copy=False)
modelC = GradientBoostingClassifier(n_estimators=150,verbose=2)
modelC = modelC.fit(traC,tarC)

#test & prediction
tsthholdC = pd.read_csv("~/PoverT/C_hhold_test.csv")
tstindivC = pd.read_csv("~/PoverT/C_indiv_test.csv")
tstindivC = tstindivC.drop(['iid','country'],axis=1)
d3 = pd.merge(tsthholdC,tstindivC,on='id',how='left')
t3 = d3.isnull().sum()
t3[t3>0]
idsC = d3.id
d3 = d3.drop(['id','country'],axis=1)
s3 = d3.dtypes
u3 = list(s3[s3==object].index)
for y in u3:
    d3[y] = d3[y].astype('category').cat.codes

tesC = np.array(d3,order='C',copy=False)
predC = modelC.predict_proba(tesC)

#submitB
subC = pd.DataFrame(idsC,columns=['id'])
subC['poor'] = predC[:,1]
ids3 = tsthholdC.id
listC = []
for i in ids3:
    listC.append(subC.loc[subC.id==i,'poor'].mean())

subCf = pd.DataFrame(ids3,columns=['id'])
subCf['country'] = 'C'
subCf['poor'] = listC

#final
sub1 = pd.concat([subAf,subBf,subCf])
sub1.to_csv('PoverT2.csv',index=False)

