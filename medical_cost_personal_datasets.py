#!/usr/bin/env python
# coding: utf-8

# # Dataset [link](https://www.kaggle.com/mirichoi0218/insurance)

# # Importing Libraries

# In[6]:


import numpy as np
import pandas as pd
import xgboost as xgb
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from prettytable import PrettyTable


# In[2]:


sklearn.__version__, xgb.__version__


# # Loading data

# In[ ]:


df = pd.read_csv("insurance.csv")
df.head()


# # Exploratory Data Analysis

# In[ ]:


df.describe()


# In[ ]:


df.info()


# * There are a total of 1338 rows and there are no null values in the dataset.

# #### age

# In[ ]:


print(f"minimum age: {df['age'].min()}, maximum age: {df['age'].max()}")
sns.histplot(df, x = "age", bins = 23, kde = True)
plt.show()


# * From the above figure, we can see that, in most of the age bin ranges, the distribution of population in age bins is almost uniform except in first and last bin.

# In[ ]:


plt.figure(figsize = (10, 5))
sns.histplot(df, x = "age", hue = "sex", multiple = "dodge", shrink = .8)
plt.show()


# * From the above figure, we can see that, in each bin, male and female are almost equal.

# In[ ]:


plt.figure(figsize = (10, 5))
sns.histplot(df, x = "age", hue = "children", multiple = "dodge", shrink = .8)
plt.show()


# In[ ]:


plt.figure(figsize = (10, 5))
sns.histplot(df, x = "age", hue = "smoker", multiple = "dodge", shrink = .8)
plt.show()


# * From the above figure, we can see that, in each bin, the population of smokers is around 1/3rd of non-smokers.

# In[ ]:


plt.figure(figsize = (10, 5))
sns.histplot(df, x = "age", hue = "region", multiple = "dodge", shrink = .8)
plt.show()


# * From the above figure, we can see that, in each bin, there are almost equal population from every region.

# #### bmi

# In[ ]:


print(f"minimum BMI: {df['bmi'].min()}, maximum BMI: {df['bmi'].max()}")
sns.displot(df, x = "bmi", kde = True)
plt.show()


# In[ ]:


plt.figure(figsize = (10, 5))
sns.displot(df, x = "bmi", hue = "sex", kind = "kde")
plt.show()


# * From the above figure, we can see that, male and female bmi distribution is similar.

# In[ ]:


plt.figure(figsize = (10, 5))
sns.displot(df, x = "bmi", hue = "sex", kind = "kde", col = "smoker")
plt.show()


# In[ ]:


plt.figure(figsize = (10, 7))
sns.displot(df, x = "bmi", hue = "sex", kind = "kde", col = "region")
plt.show()


# #### region

# In[ ]:


sns.histplot(df, x = "region")
plt.show()


# In[ ]:


plt.figure(figsize = (10, 5))
sns.histplot(df, x = "region", hue = "sex", multiple = "dodge", shrink = .8)
plt.show()


# * From the above figure, we can see that, from each region the male and female population is almost equal.

# In[ ]:


plt.figure(figsize = (10, 5))
sns.histplot(df, x = "region", hue = "smoker", multiple = "dodge", shrink = .8)
plt.show()


# * From the above figure, we can see that, in each region, the ratio of smoker to non-smoker is almost equal.

# #### charges

# In[ ]:


print(f"minimum charge: {df['charges'].min()}, maximum charge: {df['charges'].max()}")
sns.displot(df, x = "charges", kde = True)
plt.show()


# In[ ]:


plt.figure(figsize = (10, 5))
plt.scatter(df["age"], df["charges"])
plt.xlabel("age")
plt.ylabel("charges")
plt.title("charges vs age scatter plot")
plt.show()


# * From the above figure, we can see that, as the age increases the charges increases. We can see an upward trend in charges. This is obvious as the age increses the chance of getting illness increases and medical charges increses.

# In[ ]:


plt.figure(figsize = (10, 5))
sns.displot(df, x = "charges", hue = "sex", kind = "kde")
plt.show()


# In[ ]:


plt.figure(figsize = (10, 5))
sns.displot(df, x = "charges", hue = "smoker", kind = "kde")
plt.show()


# * From the above figure, we can see that, smokers pay high charges when compared to non-smokers.

# In[ ]:


plt.figure(figsize = (10, 7))
sns.displot(df, x = "charges", hue = "sex", kind = "kde", col = "region")
plt.show()


# In[ ]:


df_expanded = pd.get_dummies(df)
df_expanded.drop(columns = ["sex_male", "smoker_no"], inplace = True)
df_expanded.head()


# In[ ]:


train, test = train_test_split(df_expanded, test_size = .25)

train.shape, test.shape


# In[ ]:


x_train = train.drop(columns = ["charges"])
x_test = test.drop(columns = ["charges"])
y_train = np.array(train["charges"]).reshape(-1, 1)
y_test = np.array(test["charges"]).reshape(-1, 1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaler.fit(x_train)
y_scaler.fit(y_train)

x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


x_train_scaled.shape, x_test_scaled.shape, y_train_scaled.shape, y_test_scaled.shape


# # Linear Regression

# In[ ]:


model = ElasticNet()
params = {"alpha": [.00001, .000033, .0001, .00033, .001, .0033, .01, .033, .1, .33, 1, 3.3, 10, 33, 100], 
          "l1_ratio" : [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]}
gridsearch = GridSearchCV(model, params, scoring = "neg_mean_absolute_percentage_error", n_jobs = -1, 
                          cv = 5, return_train_score = True, verbose = 1)
gridsearch.fit(x_train_scaled, y_train_scaled)


# In[ ]:


results = pd.DataFrame.from_dict(gridsearch.cv_results_)
results["mean_test_score"] = results["mean_test_score"].apply(abs)
results["mean_train_score"] = results["mean_train_score"].apply(abs)
results.head()


# In[ ]:


hmap = results.pivot("param_alpha", "param_l1_ratio", "mean_train_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("l1_ratio")
plt.ylabel("alpha")
plt.title("mean absolute percentage error of training data in heatmap")
plt.show()


# In[ ]:


hmap = results.pivot("param_alpha", "param_l1_ratio", "mean_test_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("l1_ratio")
plt.ylabel("alpha")
plt.title("mean absolute percentage error of testing data in heatmap")
plt.show()


# In[ ]:


l1_ratio = .1
alpha = 1


# In[ ]:


model = ElasticNet(l1_ratio = l1_ratio, alpha = alpha)
model.fit(x_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

mean_squared_error(y_test, y_pred, squared = False), mean_absolute_percentage_error(y_test, y_pred)


# In[ ]:


importance = pd.DataFrame()
importance["feature_name"] = train.drop(columns = ["charges"]).columns
importance["coefficient"] = model.coef_
importance["sbs_coefficient"] = importance["coefficient"].apply(abs)
importance.sort_values(by = ["sbs_coefficient"], ascending = False, ignore_index = True, inplace = True)
importance


# * From feature importance, we can see that, a smoker pays more charges and has highest influence on medical charges. Intuitively, there is a higher probability that a smoker has health issues and goes to hospital, so medical charges increases.
# * As age increases, the probability of getting health issues increases, so this effects medical charges.
# * BMI also has an effect on charges as observed from feature importance.
# * Important thing to note is our model doesn't have bias towards gender, number of children and region.

# # kNN Regressor

# In[ ]:


model = KNeighborsRegressor()
params = {"n_neighbors" : [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45], 
          "weights" : ["uniform", "distance"]}
gridsearch = GridSearchCV(model, params, scoring = "neg_mean_absolute_percentage_error", n_jobs = -1, 
                          cv = 5, return_train_score = True, verbose = 1)
gridsearch.fit(x_train_scaled, y_train_scaled)


# In[ ]:


results = pd.DataFrame.from_dict(gridsearch.cv_results_)
results["mean_test_score"] = results["mean_test_score"].apply(abs)
results["mean_train_score"] = results["mean_train_score"].apply(abs)
results.head()


# In[ ]:


hmap = results.pivot("param_n_neighbors", "param_weights", "mean_train_score")
plt.figure(figsize = (5, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.ylabel("n_neighbors")
plt.xlabel("weights")
plt.title("mean absolute percentage error of training data in heatmap")
plt.show()


# In[ ]:


hmap = results.pivot("param_n_neighbors", "param_weights", "mean_test_score")
plt.figure(figsize = (5, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("weights")
plt.ylabel("n_neighbors")
plt.title("mean absolute percentage error of testing data in heatmap")
plt.show()


# In[ ]:


n_neighbors = 17
weights = "distance"


# In[ ]:


model = KNeighborsRegressor(n_neighbors = n_neighbors, weights = weights)
model.fit(x_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

mean_squared_error(y_test, y_pred, squared = False), mean_absolute_percentage_error(y_test, y_pred)


# # SVM Regressor

# In[ ]:


model = SVR()
params = {"kernel" : ["linear", "poly", "rbf", "sigmoid"], 
          "C" : [.00001, .000033, .0001, .00033, .001, .0033, .01, .033, .1, .33, 1, 3.3, 10, 33, 100]}
gridsearch = GridSearchCV(model, params, scoring = "neg_mean_absolute_percentage_error", n_jobs = -1, 
                          cv = 5, return_train_score = True, verbose = 1)
gridsearch.fit(x_train_scaled, y_train_scaled)


# In[ ]:


results = pd.DataFrame.from_dict(gridsearch.cv_results_)
results["mean_test_score"] = results["mean_test_score"].apply(abs)
results["mean_train_score"] = results["mean_train_score"].apply(abs)
results.head()


# In[ ]:


hmap = results.pivot("param_C", "param_kernel", "mean_train_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("kernel")
plt.ylabel("C")
plt.title("mean absolute percentage error of training data in heatmap")
plt.show()


# In[ ]:


hmap = results.pivot("param_C", "param_kernel", "mean_test_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("kernel")
plt.ylabel("C")
plt.title("mean absolute percentage error of testing data in heatmap")
plt.show()


# In[ ]:


C = 1
kernel = "poly"


# In[ ]:


model = SVR(C = C, kernel = kernel)
model.fit(x_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

mean_squared_error(y_test, y_pred, squared = False), mean_absolute_percentage_error(y_test, y_pred)


# # Random Forest Regressor

# In[ ]:


model = RandomForestRegressor(max_depth = None)
params = {"n_estimators" : [1, 5, 10, 25, 50, 75, 100], 
          "max_features" : ["auto", None, "sqrt", "log2"]}
gridsearch = GridSearchCV(model, params, scoring = "neg_mean_absolute_percentage_error", n_jobs = -1, 
                          cv = 5, return_train_score = True, verbose = 1)
gridsearch.fit(x_train_scaled, y_train_scaled)


# In[ ]:


results = pd.DataFrame.from_dict(gridsearch.cv_results_)
results["mean_test_score"] = results["mean_test_score"].apply(abs)
results["mean_train_score"] = results["mean_train_score"].apply(abs)
results.head()


# In[ ]:


hmap = results.pivot("param_n_estimators", "param_max_features", "mean_train_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("max_features")
plt.ylabel("n_estimators")
plt.title("mean absolute percentage error of training data in heatmap")
plt.show()


# In[ ]:


hmap = results.pivot("param_n_estimators", "param_max_features", "mean_test_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("max_features")
plt.ylabel("n_estimators")
plt.title("mean absolute percentage error of testing data in heatmap")
plt.show()


# In[ ]:


max_features = "log2"
n_estimators = 75


# In[ ]:


model = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features)
model.fit(x_train_scaled, y_train_scaled.ravel())

y_pred_scaled = model.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

mean_squared_error(y_test, y_pred, squared = False), mean_absolute_percentage_error(y_test, y_pred)


# In[ ]:


importance = pd.DataFrame()
importance["feature_name"] = train.drop(columns = ["charges"]).columns
importance["coefficient"] = model.feature_importances_
importance.sort_values(by = ["coefficient"], ascending = False, ignore_index = True, inplace = True)
importance


# * From feature importance, we can see that, a smoker pays more charges and has highest influence on medical charges. Intuitively, there is a higher probability that a smoker has health issues and goes to hospital, so medical charges increases.
# * As age increases, the probability of getting health issues increases, so this effects medical charges.
# * BMI also has an effect on charges as observed from feature importance.
# * Important thing to note is our model has a very small bias towards gender and region when compared to smoker.

# # XGBoost Regressor

# In[ ]:


model = xgb.XGBRegressor()
params = {"n_estimators" : [1, 5, 10, 25, 50, 75, 100], 
          "max_depth" : [4, 8, 16, 32]}
gridsearch = GridSearchCV(model, params, scoring = "neg_mean_absolute_percentage_error", n_jobs = -1, 
                          cv = 5, return_train_score = True, verbose = 1)
gridsearch.fit(x_train_scaled, y_train_scaled)


# In[ ]:


results = pd.DataFrame.from_dict(gridsearch.cv_results_)
results["mean_test_score"] = results["mean_test_score"].apply(abs)
results["mean_train_score"] = results["mean_train_score"].apply(abs)
results.head()


# In[ ]:


hmap = results.pivot("param_n_estimators", "param_max_depth", "mean_train_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("max_depth")
plt.ylabel("n_estimators")
plt.title("mean absolute percentage error of training data in heatmap")
plt.show()


# In[ ]:


hmap = results.pivot("param_n_estimators", "param_max_depth", "mean_test_score")
plt.figure(figsize = (10, 10))
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("max_depth")
plt.ylabel("n_estimators")
plt.title("mean absolute percentage error of testing data in heatmap")
plt.show()


# In[ ]:


n_estimators = 50
max_depth = 4


# In[ ]:


model = xgb.XGBRegressor(n_estimators = n_estimators, max_depth = max_depth)
model.fit(x_train_scaled, y_train_scaled.ravel())

y_pred_scaled = model.predict(x_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

mean_squared_error(y_test, y_pred, squared = False), mean_absolute_percentage_error(y_test, y_pred)


# In[ ]:


importance = pd.DataFrame()
importance["feature_name"] = train.drop(columns = ["charges"]).columns
importance["coefficient"] = model.feature_importances_
importance.sort_values(by = ["coefficient"], ascending = False, ignore_index = True, inplace = True)
importance


# * From feature importance, we can see that, a smoker pays more charges and has highest influence on medical charges. Intuitively, there is a higher probability that a smoker has health issues and goes to hospital, so medical charges increases.
# * As age increases, the probability of getting health issues increases, so this effects medical charges.
# * BMI also has an effect on charges as observed from feature importance.
# * Important thing to note is our model has a very very small bias towards gender and region when compared to smoker.

# In[9]:


p = PrettyTable(["Model", "test MAPE"])
p.add_row(["Linear Regression with L1 \nand L2 regularization", "0.9429"])
p.add_row(["kNN Regressor", "0.3386"])
p.add_row(["SVM Regressor", "0.2112"])
p.add_row(["Random Forest Regressor", "0.3064"])
p.add_row(["XGBoost Regressor", "0.2663"])
print(p)


# In[ ]:




