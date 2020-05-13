import pandas as pd
import lightgbm as lg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
trainData = pd.read_csv("C:/Users/rroutr01/Downloads/train_NA17Sgz/train.csv")
testData = pd.read_csv("C:/Users/rroutr01/Downloads/test_aq1FGdB/test.csv")
# checking missing values
trainData.isnull().sum(axis=0)/trainData.shape[0]

#impression_id      0.0
#impression_time    0.0
#user_id            0.0
#app_code           0.0
#os_version         0.0
#is_4G              0.0
#is_click           0.0
#dtype: float64 

cols = ['impression_time','user_id','app_code','os_version','is_4G']
#categorize other fields
for col in cols:
      lbl = LabelEncoder()
      lbl.fit(list(trainData[col].values) + list(testData[col].values))
      trainData[col] = lbl.transform(list(trainData[col].values))
      testData[col] = lbl.transform(list(testData[col].values))

trainData.head()

#                      impression_id      impression_time  user_id  app_code    os_version  is_4G  is_click
#0  c4ca4238a0b923820dcc509a6f75849b  2018-11-15 00:00:00    87862       422           old      0         0
#1  45c48cce2e2d7fbdea1afc51c7c6ad26  2018-11-15 00:01:00    63410       467        latest      1         1
#2  70efdf2ec9b086079795c442636b55fb  2018-11-15 00:02:00    71748       259  intermediate      1         0
#3  8e296a067a37563370ded05f5a3bf3ec  2018-11-15 00:02:00    69209       244        latest      1         0
#4  182be0c5cdcd5072bb1864cdee4d3d6e  2018-11-15 00:02:00    62873       473        latest      0         0

columns_to_use = list(set(trainData.columns) - set(['impression_id','is_click']))

X_train, X_test, y_train, y_test = train_test_split(trainData[columns_to_use], trainData['is_click'], test_size = 0.5)
# applying lightgbm algorithm
dtrain = lg.Dataset(X_train, y_train)
dvalue = lg.Dataset(X_test, y_test)
parameters = {
     'num_leaves' : 200,
     'learning_rate':0.03,
     'metric':'auc',
     'objective':'binary',
     'early_stopping_round': 50,
     'max_depth':10,
     'bagging_fraction':0.5,
     'feature_fraction':0.6,
     'bagging_seed':2017,
     'feature_fraction_seed':2017,
     'verbose' : 1
}
cf = lg.train(parameters, dtrain,num_boost_round=500,valid_sets=dvalue,verbose_eval=20)
predict = cf.predict(testData[columns_to_use])
subs = pd.DataFrame({'impression_id':testData['impression_id'], 'is_click':predict})
# writing output
subs.to_csv('C:/Users/rroutr01/Downloads/sample_submission_IPsBlCT/sample_submission.csv', index=False)


