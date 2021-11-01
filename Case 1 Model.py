#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyreadr
import pandas as pd 
import numpy as np

import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime as dt
import datetime
import timeit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


# In[2]:


data = pd.read_csv('loans_full_schema.csv')


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.dtypes


# In[6]:


data.head(5)


# In[7]:


data.describe()


# Explore the data and figure out issues

# In[8]:


# explainatory analysis

# a function to generate basic info about the field, and examine the data records

def basic(field):
    display("# of Zeros:", data[field].isna().sum())
    print(" ")
    display("Uniqueness:", data[field].unique())
    display("# of Unique:", len(data[field].unique()))
    print(" ")
    display("value counts:", data[field].value_counts())
    print(" ")
    display("# of Records", data[field].shape[0] - data[field].isna().sum())


# In[9]:


# examine data issue - missing
# find out no. of records with 0 and the percentage of such records

def zero(field):
    display("# of Zeros:", data[field].isna().sum())
    print(" ")
    display("% of Zeros:", data[field].isna().sum()/data[field].shape[0])
    
field = data.columns
zero(field = field)


# Missing information:
# 
# The data has significantly missing information on below aspects, we will drop the columns with more than 50% records missing. Even if we try to fill in the information, it may not be accurate as the majority is missing. 
# 1. Employment information - employment title and length both have around 800 records missing, but the missing rate is not that high. 
# 2. Joint application - for joint applications, there are 3 columns indicating the annual_income, verification_income and debt_to_income. These 3 columns are basically empty. As the missing percentage is above 85%, we will drop these columns before modeling. 
# 3. Recency, months_since_last_delinq, months_since_90d_late,these 2 columns also see significant missing. We will also drop them. 
# 4. debt_to_income, months_since_last_credit_inquiry, num_accounts_120d_past_due, these 3 columns' missing is not as much as joint applications'. Therefore we will try to fill in the missing information. 

# In[10]:


# Drop columns with too many missing records(missing > 50%)

data = data.drop(columns = ['annual_income_joint', 'verification_income_joint', 'debt_to_income_joint','months_since_90d_late', 'months_since_last_delinq'])


# In[11]:


# Fill in missing information 

# Filling In Missing debt_to_income, months_since_last_credit_inquiry, and num_accounts_120d_past_due
# calculate groupwise average. First replace the 0’s and 1’s by NAs so they are not counted in calculating mean.

data.loc[data['debt_to_income']==0,'debt_to_income']=np.nan
data.loc[data['months_since_last_credit_inquiry']==0,'months_since_last_credit_inquiry']=np.nan
data.loc[data['num_accounts_120d_past_due']==0,'num_accounts_120d_past_due']=np.nan

# calculate the mean now (mean function ignores NAs but not 0s hence we converted 0 to NA)
mean_debt_to_income = data['debt_to_income'].mean()
mean_months_since_last_credit_inquiry = data['months_since_last_credit_inquiry'].mean()
mean_num_accounts_120d_past_due= data['num_accounts_120d_past_due'].mean()

# impute values
data.loc[(data['debt_to_income'].isnull()),'debt_to_income']=mean_debt_to_income   
data.loc[(data['months_since_last_credit_inquiry'].isnull()),'months_since_last_credit_inquiry']=mean_months_since_last_credit_inquiry
data.loc[(data['num_accounts_120d_past_due'].isnull()),'num_accounts_120d_past_due']=mean_num_accounts_120d_past_due
    
data["emp_length"] = data["emp_length"].fillna(data["emp_length"].mean())


# In[12]:


# examine data issue - duplication 

data[data.duplicated()].count()


# No duplicated records found. 

# In[13]:


# examine data issue - outlier 

plt.figure(figsize=(9, 16), dpi=300)
data.boxplot(fontsize=6)
plt.ticklabel_format(axis='y', style='plain')
plt.xticks(rotation=90)
plt.show()


# We can see from the chart that:
# Annual_income,total_credit_limit,total_credit_utilized,total_collection_amount_ever,total_debit_limit have outliers. But all these factors indicate the general financial habits and economic status - income level, credit score, credit card usage, etc. All these factors naturally tend to have outliers as they vary from customer to customer, therefore we consider these outliers acceptable and won't exclude them. 

# In[14]:


# transform data type 
# transform all object into string, but time into datetime

entities = ["emp_title","state","verified_income","loan_purpose","application_type","grade","sub_grade","loan_status","initial_listing_status","disbursement_method","verified_income","homeownership","state"]
data[entities] = data[entities].astype("str")

data['issue_month']=pd.to_datetime(data['issue_month'])


# Data visualisation and analysis

# In[15]:


# Visualisation - revenue - paid total

field = 'paid_total'
basic(field = field)

# plot to see distribution
plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')

plot.set_title('Paid Total Top30')

plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# In[36]:


#paid_total_distribution

plt.figure(figsize=(4, 2.5), dpi=300)
plt.tick_params(labelsize=8)
sns.distplot(data["paid_total"])
plt.title("Paid Total Distribution",fontsize=8)
plt.xlabel("Paid Total", fontsize=8)
plt.ylabel("")
plt.show()


# In[33]:


# revenue - paid interest

field = 'paid_interest'
basic(field = field)

# plot to see distribution

plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')
plot.set_title('Paid Interest Top30')
plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# In[34]:


# paid-interest distribution

plt.figure(figsize=(16, 9), dpi=300)
plt.tick_params(labelsize=8)
sns.distplot(data["paid_interest"])
plt.title("Paid Interest Distribution",fontsize=8)
plt.xlabel("Paid Interest", fontsize=8)
plt.ylabel("")
plt.show()


# Revenue analysis: 
# Most paid_totals are 0, which indicates that, either most of the loans have just been released, or we are missing information. Here we selected top 30 paid totals, most of them share a range of [900, 1600].
# Similarly, most paid interests are 0 as loans were just released or missing information. Among top 30 paid interests, most of them share a range of 
# Both paid total and interest are significantly skewed, the frequency decreases as the amount increases, which means smaller amount loans and interests are m

# In[17]:


# loan issue distributed by time

data.groupby('issue_month')['paid_total'].sum().plot(title='Paid-Total Each Month',kind='line',figsize=[16,9])


# In[18]:


# revenue geographic distribution - paid-total & paid-interest distributed by state

field = 'state'
basic(field = field)

#plot 

#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:5]

#create pie chart
data.groupby('state')['paid_total'].sum().head(7).plot(title='Paid-Total by State',kind='pie',figsize=[16,9], autopct='%.0f%%')


# In[19]:


data.groupby('state')['paid_interest'].sum().head(7).plot(title='Paid-Interest by State',kind='pie',figsize=[16,9], autopct='%.0f%%')


# The source of paid_total and paid_interest highly overlap with each other. Most of our revenue comes from loan in CA, AZ, CO, CT these 4 states. 
# Possible Reasons: more loan applicants from these states; larger loan amount or higher interest rate.
# Suggestions: assign more investigator, or customer relation employees in these states to follow up the loan applicants. 

# In[20]:


# customer profile - income

field = 'annual_income'
basic(field = field)

# plot to see distribution

plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')

plot.set_title('Annual Income Distribution')

plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# In[21]:


# customer profile - homeownship

field = 'homeownership'
basic(field = field)

# plot to see distribution

plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')

plot.set_title('Homeownership Distribution')

plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# In[22]:


# customer profile - debt to income ratio

field = 'debt_to_income'
basic(field = field)

# plot to see distribution

plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')

plot.set_title('Debt-to-income ratio distribution')

plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# Customer Profile: 
# Most of our loan applicants are having a mortgage or renting homes. Their income distribution is significantly skewed, and most annual incomes are between 55000 - 700000. Consequently, the debt_to_income ratio also has a skewed distribution, most of the debt_to_income ratios fall between 12 to 16. 

# In[23]:


# risk analysis - loan grade 

field = 'grade'
basic(field = field)

# plot to see distribution

plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')

plot.set_title('Loan Grade Distribution')

plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# In[24]:


# risk analysis - historical failure to pay

field = 'num_historical_failed_to_pay'
basic(field = field)

# plot to see distribution

plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')

plot.set_title('Historical Failures to Pay')

plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# In[25]:


# risk analysis - delinquent accounts

field = 'current_accounts_delinq'
basic(field = field)

# plot to see distribution

plt.rcParams["figure.figsize"] = [16,9]
sns.set(font_scale = 2, style='whitegrid')

colors = sns.color_palette('pastel')[9]

plot = sns.countplot(x=field, 
                     data=data, 
                     order=data[field].value_counts().iloc[:30].index, 
                     color=colors)

plot.set_yscale('log')

plot.set_title('Current Delinquent Accounts')

plot.set_xticklabels(plot.get_xticklabels(),rotation=45)


# Risk analysis 
# Viewing from the behavior of our customers, the majority of them don't have any delinquent accounts or historical failure to pay. Most historical failures to pay are concentrated between 1 to 3 times. Meanwhile, most of the loans' grades are B, A, C. These 2 factors indicate that we can have an optimistic assumption - the risk of failing to pay back is relatively low. 
# To identify risk, especially quantitatively predict the risk and paid-interest, we need to consider many other factors, which will be further explored in models. 

# Modeling: as paid_interest rate is what we will predict, that means paid_interest is the dependent variable. With dependent variable, we will build supervised models, 2 models we suggest here are neural network and random forest as they tend to be both accurate and explainable. To build models, we need to conduct feature selection to see which factors we will include in the models. 

# In[26]:


# prep data, convert data types

def str_to_digit(df, col):
    map_dict = {}
    count = 1
    for value in df[col].unique():
        map_dict[value] = count
        count += 1
    return df[col].map(map_dict)

remove_cols = ["emp_title", "state", "num_accounts_120d_past_due", "issue_month", "paid_interest"]
for col in data.columns:
    if col not in remove_cols and data[col].dtypes == "O":
        data[col] = str_to_digit(data, col)

data["loan_year_digit"] = data["issue_month"].dt.year
data["loan_month_digit"] = data["issue_month"].dt.month


# In[27]:


# Feature Selection

select_features = []

for col in data.columns:
    if col not in remove_cols:
        corr, p_value = stats.pearsonr(data[col], data['paid_interest'])
        if abs(corr) > 0.1 and p_value < 0.001:
            select_features.append(col)

select_data = data[select_features + ["paid_interest"]]

plt.figure(figsize=(16, 9), dpi=300)
plt.tick_params(labelsize=8)
plt.title("Correlationships Among Important Features",fontsize=8)
sns.heatmap(round(select_data.corr(), 2), annot=True, cmap="Blues", annot_kws={"fontsize":3})
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=4)
plt.show()


# In[28]:


# Split train/test dataset

X, y = select_data.iloc[:,:-1], select_data["paid_interest"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[29]:


# Decision Tree
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
tree_rmse = np.sqrt(mse(y_pred, y_test))
tree_mae = mae(y_pred, y_test)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mse(y_pred, y_test))
rf_mae = mae(y_pred, y_test)

# Results & Measure of Goodness
result = pd.DataFrame({
    "Model": ["DecisionTree", "RandomForest"],
    "RMSE": [tree_rmse, rf_rmse],
    "MAE": [tree_mae, rf_mae]
})

result


# In[30]:


# Visualise the results

plt.figure(figsize=(9, 16), dpi=300)
plt.tick_params(labelsize=8)
plt.bar([0.8, 1.8], [tree_rmse, tree_mae], width=0.4, label="Decision Tree")
plt.bar([1.2, 2.2], [rf_rmse, rf_mae], width=0.4, label="Random Forest")
plt.title("Results for Decision Tree and Random Forest Model",fontsize=7)
plt.xticks([1,2], ["RMSE", "MAE"])
plt.legend(fontsize=5)
plt.show()


# In[ ]:




