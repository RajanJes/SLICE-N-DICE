# -*- coding: utf-8 -*-
"""
Created on Fri May 24

@author: ROMY
"""
# =============================================================================
# CLASSIFYING PERSONAL INCOME 
# =============================================================================
################################# Required packages ############################
# to work with dataframes
import pandas as pd 

# to perform numerical operations
import numpy as np

# to visualize data
import seaborn as sns

# to partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

###############################################################################
# =============================================================================
# Importing data and dummy conversion
# =============================================================================
data_income = pd.read_csv('income.csv')                                         #,na_values=[" ?"]) 
                                                                                # Additional strings (" ?") to recognize as NA
data = data_income.copy()

"""
#Exploratory data analysis:

#1.Data information (types and missing value) and statistics
#2.Data preprocessing (Missing values, handling categorical variables etc)
"""
# =============================================================================
# Data Information
# =============================================================================
print(data.info())             ## To check variable's data type
data.isnull()                  ## Check for missing values
print('Data columns with null values:\n', data.isnull().sum()) ## No missing values !

summary_num = data.describe()
print(summary_num)             ## For numerical variables

summary_cate = data.describe(include = "O") ## For categorical variables
"""
Checking for unique classes
"""
print(np.unique(data['JobType'])) ## There exists ' ?' instesd of nan
print(np.unique(data['occupation']))  ## There exists ' ?' instesd of nan

"""
Go back and read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
"""
data = pd.read_csv('income.csv',na_values=[" ?"]) 

# =============================================================================
# Data pre-processing
# =============================================================================
data.isnull().sum()
missing = data[data.isnull().any(axis=1)]
# any axis=1 => to check for least one missing value per row

# filter the rows with missing values 
# to know about the rows with missing values
missing_job= data[(data['JobType'].isnull())]
missing_occ= data[(data['occupation'].isnull())]

# Points to note:
#*** 1. Missing values in Jobtype    = 1809
#    2. Missing values in Occupation = 1816 
#    3. There are 1809 rows where two specific 
#       columns i.e. occupation & JobType have missing values
#    4. (1816-1809) = 7 => You still have occupation unfilled for 
#***    these 7 rows. Because, jobtype is Never worked

data.nativecountry.value_counts(sort=True) 
# only one row with Holand-Netherlands

data.JobType.value_counts(sort=True)
# Without-pay and Never-worked will anyways fall under <=50000
# It is equivalent of not having it in the dataframe. Let's remove them

data2 = data[(data['nativecountry'] !=' Holand-Netherlands')]
data2 = data2[(data2['JobType'] !=' Without-pay')]
data2 = data2[(data2['JobType'] !=' Never-worked')]

# =============================================================================
# Missing Value Imputation
# =============================================================================
# JobType
# checking the frequencie of categories
data.JobType.value_counts(sort=True)
 
# Filling the missing value with modal value
data2.JobType.mode()
data2.JobType.fillna(data2.JobType.mode()[0],inplace=True)
# use [0] to replace mising values with the most frequently occuring category
data2.isnull().sum()

# Occupation
data2.occupation.value_counts(sort=True)
data2.occupation.fillna(data2.occupation.mode()[0],inplace=True)
# Checking if there are any missing values left 
data2.isnull().sum()

# =============================================================================
# Data Visualization & Cross tables
# =============================================================================
data2.columns                 # Extracting the column names
# 1. Salary Status - Frequency Table
gender = pd.crosstab(index=data2["gender"], columns='count')
print(gender)

# Plotting the gender variable
gender_plot = sns.countplot(data['gender'])
## The data consists of 75 % people whose salary status is <=50,000 
## & 25% people whose salary status is > 50,000

###############################################################################
#2. Age 
##############  Histogram of Age  #############################
sns.distplot(data['age'], bins=10, kde=True)
# People with age 20-45 age are high in frequency

############# Box Plot - Age vs Salary status #################
sns.boxplot('SalStat', 'age', data=data)
data.groupby('SalStat')['age'].median()
## people with 30-50 age are more likely to fall under > 50000 salary status
## people with 25-35 age are more likely to fall under <= 50000 salary status

# =============================================================================
# Try out Data Visualization and EDA for other numerical variables!
# 'capitalgain', 'capitalloss', 'hoursperweek'
# =============================================================================

# 3. gender 
# Plotting the outcome variable - gender status
sns.countplot(data['gender'])
# male frequency is high

# 4. age vs hoursperweek
sns.regplot(x='age', y='hoursperweek', fit_reg = False, data=data)

# =============================================================================
# One-way table:
# =============================================================================
gender = pd.crosstab(index=data["gender"], 
                     columns='count')
print(gender)
# =============================================================================
#  Two-way table:
# =============================================================================
gender2 = pd.crosstab(index=data["gender"],
                     columns=data['SalStat'],
                     margins = True)  # Include row and column totals
print(gender2)

gender2.columns = [' Female', ' Male', 'rowtotal']
gender2.index   = [' Female', ' Male', 'coltotal']
print(gender2)

# =============================================================================

# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the values from input features
x = new_data[features].values
print(x)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)


# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)

# Prediction from test data
prediction = logistic.predict(test_x)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

# =============================================================================
# END OF SCRIPT
# =============================================================================
