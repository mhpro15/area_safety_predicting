
import pandas as pd
data_group1=pd.read_csv('project_data.csv')

data_group1.head(5)
print(data_group1.columns.values)

for col in data_group1.columns:
    print(col)

data_group1.describe()

data_group1.fillna("missing",inplace=True)
data_group1.head(30)

data_group1 = data_group1.drop(['INDEX', 'ObjectId'], axis=1, errors='ignore')
data_group1.head(5)

data_group1 = data_group1[data_group1['NEIGHBOURHOOD'] != 'NSA']
data_group1 = data_group1[data_group1['DIVISION'] != 'NSA']
data_group1['CRIME_TYPE'] = data_group1['CATEGORY'] + "-" + data_group1['SUBTYPE']
data_group1['Percentage_Clear'] = (data_group1['COUNT_CLEARED']/data_group1['COUNT']) * 100
data_group1.tail(30)

data_group1['NEIGHBOURHOOD'] = data_group1['NEIGHBOURHOOD'].str.replace(r'\D', '', regex=True)
data_group1['DIVISION'] = data_group1['DIVISION'].str.strip('D')

crime_types_list = data_group1['CRIME_TYPE'].unique()

import numpy as np

for i in range(len(crime_types_list)):
  data_group1['CRIME_TYPE']=np.where(data_group1['CRIME_TYPE'] == crime_types_list[i], i+1, data_group1['CRIME_TYPE'])


data_group1.tail(30)

data_group1 = data_group1.drop(['CATEGORY', 'SUBTYPE'], axis=1, errors='ignore')
data_group1.tail(30)

import matplotlib.pyplot as plt

yearly_counts = data_group1.groupby('REPORT_YEAR').agg({'COUNT': 'sum', 'COUNT_CLEARED': 'sum'})

plt.plot(yearly_counts.index, yearly_counts['COUNT'], label='COUNT')
plt.plot(yearly_counts.index, yearly_counts['COUNT_CLEARED'], label='COUNT_CLEARED')

plt.xlabel('REPORT_YEAR')
plt.ylabel('Count')
plt.title('Counts Over Time')
plt.legend()
plt.show()

yearly_percentage = data_group1.groupby('REPORT_YEAR').agg({'Percentage_Clear':'mean'})

plt.plot(yearly_percentage.index, yearly_percentage['Percentage_Clear'], label='Percentage_Clear')

plt.xlabel('REPORT_YEAR')
plt.ylabel('Pecentage_Clear')
plt.title('Percentage Clear Over Time')
plt.legend()
plt.show()

data_group1['Safe'] = (data_group1['Percentage_Clear'] > 47.6).astype(int)
data_group1['CLEAR_RATE'] = pd.cut(data_group1['Percentage_Clear'], bins=[-1, 20, 40, 60, 80, 100], labels=[1, 2, 3, 4, 5])
data_group1.tail(30)

data_group1_vars=data_group1.columns.values.tolist()
data_group1['Safe'] = data_group1['Safe'].astype(int)
Y=['Safe']
X=[i for i in data_group1_vars if i not in Y ]
type(Y)
type(X)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, n_features_to_select=3)
rfe = rfe.fit(data_group1[X], data_group1[Y])
print(rfe.support_)
print(rfe.ranking_)

cols=['CRIME_TYPE', 'REPORT_YEAR','NEIGHBOURHOOD','DIVISION']
X=data_group1[cols]
Y=data_group1['Safe']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn import linear_model
from sklearn import metrics
clf1 = linear_model.LogisticRegression(solver='lbfgs',max_iter=40000)
clf1.fit(X_train, Y_train)

probs = clf1.predict_proba(X_test)
print(probs)

predicted = clf1.predict(X_test)
print (predicted)

print (metrics.accuracy_score(Y_test, predicted))

import seaborn as sns
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y_test, predicted)
print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.show()

cat_vars=['CRIME_TYPE', 'DIVISION','REPORT_YEAR','NEIGHBOURHOOD']
for var in cat_vars:
  cat_list='var'+'_'+var
  print(cat_list)
  cat_list = pd.get_dummies(data_group1[var], prefix=var)
  data_group1_b=data_group1.join(cat_list)
  data_group1=data_group1_b
to_clear=['CRIME_TYPE', 'DIVISION','REPORT_YEAR','NEIGHBOURHOOD','Percentage_Clear','COUNT_CLEARED',
          'CLEAR_RATE']
data_group1_b_vars=data_group1.columns.values.tolist()
to_keep=[i for i in data_group1_b_vars if i not in to_clear]
data_group1_b_final=data_group1[to_keep]
data_group1_b_final.columns.values
print(data_group1_b_final.head(5))
# 3- Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
data_group1_b_final_vars=data_group1_b_final.columns.values.tolist()
Y=['Safe']
X=[i for i in data_group1_b_final_vars if i not in Y ]
type(Y)
type(X)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=40000)
rfe = RFE(model, n_features_to_select=3)
rfe = rfe.fit(data_group1_b_final[X],data_group1_b_final[Y] )
print(rfe.support_)
print(rfe.ranking_)
type(Y)
type(X)

from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import metrics

X = data_group1_b_final[X]
Y = data_group1_b_final[Y]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
Y_train = Y_train.values.ravel()
Y_test = Y_test.values.ravel()

clf1 = linear_model.LogisticRegression(solver='lbfgs')

from sklearn.preprocessing import StandardScaler
sScaler = StandardScaler()
X_train = sScaler.fit_transform(X_train)
X_test = sScaler.transform(X_test)
clf1.fit(X_train, Y_train)

probs = clf1.predict_proba(X_test)
print(probs)
predicted = clf1.predict(X_test)
print (predicted)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs',max_iter=40000), X, Y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())

from sklearn.tree import DecisionTreeClassifier

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
clf2 = DecisionTreeClassifier(criterion='entropy',max_depth=23, min_samples_split=100,
random_state=99)
clf2.fit(X_train, Y_train)
probs = clf2.predict_proba(X_test)
print(probs)
predicted = clf2.predict(X_test)
print (predicted)
print (metrics.accuracy_score(Y_test, predicted))
from sklearn.tree import export_graphviz
colnames=data_group1.columns.values.tolist()
predictors=colnames[:205]
with open('./dtree2.dot', 'w') as dotfile:
    export_graphviz(clf2, out_file = dotfile, feature_names = predictors)
dotfile.close()

# from sklearn.ensemble import RandomForestClassifier


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# Y_train = Y_train.values.ravel()
# Y_test = Y_test.values.ravel()


# clf3 = RandomForestClassifier(n_estimators=100, max_depth=35, random_state=0)
# clf3.fit(X_train, Y_train)
# probs = clf3.predict_proba(X_test)
# predicted = clf3.predict(X_test)
# print (metrics.accuracy_score(Y_test, predicted))


import seaborn as sns
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y_test, predicted)
print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.show()

import joblib
joblib.dump(clf2, './model.pkl')
print("Model dumped!")

model_columns = list(X.columns)
print(model_columns)
joblib.dump(model_columns, './model_columns.pkl')
print("Model columns dumped!")

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if clf2:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)

            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            scaled_df = scaler.fit_transform(query)
            # return to data frame
            query = pd.DataFrame(scaled_df, columns=model_columns)
            print(query)
            prediction = list(clf2.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    clf2 = joblib.load('./model.pkl')
    print("Model Loaded")
    model_columns = joblib.load('./model_columns.pkl')
    print("Models columns loaded")

    app.run(port=port)