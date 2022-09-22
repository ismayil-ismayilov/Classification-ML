#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project | Classification | Ismayil Ismayilov

# In[6]:


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

import seaborn as sns
import matplotlib.pyplot as plt


# ### Base function to show the Balanced Accuracy Score, the Classification Report and Confusion Matrix

# In[7]:


def Results(y_true, y_pred):
    print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_true, y_pred)}')

    print(classification_report(y_true, y_pred))

    ax = plt.subplot()

    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='g', ax=ax)

    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    plt.show()


# ## Reading the Data

# Please, input Data's location below.

# In[8]:


drugs_df = pd.read_csv('drugs_train.csv')


# In[9]:


drugs_df.head()


# Below we see every variable has 1500 observation; hence no missing observartions.

# In[10]:


drugs_df.info()


# ## Exploratory Data Analysis
# 
# Personality trait variables are all in a range of [0-100] without an outlier.

# In[11]:


drugs_df.describe()


# ## Key Stats from SDA:
# 
# 1. Genderwise, dataset is balanced: 50-50% | Male&Female
# 2. Data mostly is based on 18-34 years old as they takes more than 50% of the dataset.
# 3. 91.5% of the people are Mixed-Black/Asian
# 4. The US and Australia takes 84% of the data.
# 
# ###### Note: Except the gender, Age, Race and Country are not balanced, so it may have adverse effects in models efficiency to predict. For example, the model have lower accuracy in prediction for someone who is White, aged 55-64, from Canada

# In[12]:


print(drugs_df['gender'].value_counts(normalize=True, sort=True) * 100)


# In[13]:


drugs_df['age'].value_counts()


# In[14]:


print(drugs_df['consumption_alcohol'].value_counts(normalize=True, sort=True) * 100)


# In[15]:


print(drugs_df['education'].value_counts(normalize=True, sort=True) * 100)


# In[16]:


print(drugs_df['ethnicity'].value_counts(normalize=True, sort=True) * 100)


# In[17]:


print(drugs_df['country'].value_counts(normalize=True, sort=True) * 100)


# ### Drug Consumption

# The data includes three types of drugs: Amphetamins, Cannabis, Mushrooms. Based on the data:
# 
# 1. Around half of the people never used Amphetamine and Mushrooms, where only 21.6% people never used Cannabis.
# 2. Cannabis usage data is well balanced compared to other drug usage data.

# In[18]:


drugs_df['consumption_amphetamines'].value_counts(normalize=True, sort=True) * 100


# In[19]:


drugs_df['consumption_cannabis'].value_counts(normalize=True, sort=True) * 100


# In[20]:


drugs_df['consumption_mushrooms'].value_counts(normalize=True, sort=True) * 100


# ### Education

# The Education data reflects nearly reality. Around 50% had attained a diploma. Where 27% has an education in some college or university, but no certificate or degree.

# In[21]:


drugs_df['education'].value_counts(normalize=True, sort=True) * 100


# ## Feature engineering

# In Feature Engineering process we add:
# 1. A new feature to track "consumed illicit drugs" observations: Binary: 1 Yes, 0 No.
# 2. Features to track total number of used/taken drugs (excluding chocolate and caffeine) in different periods.
# 

# In[22]:


def Feature(df: pd.DataFrame):
    df = df.assign(no_illicit_drugs=((df[['consumption_amphetamines', 'consumption_cannabis', 'consumption_mushrooms']] == 'never used').sum(axis=1) == 3).astype(int))
  
    # TAD stands for Total Amount of Drugs consumed
    periods = [('used in last day', 'TAD_last_day'),
               ('used in last week', 'TAD_last_week'),
               ('used in last month', 'TAD_last_month'),
               ('used in last year', 'TAD_last_year'),
               ('used in last decade', 'TAD_last_decade'),
               ('used over a decade ago', 'TAD_decade_ago')]

    for period, column_name in periods:
        df = df.assign(**{column_name: (df[['consumption_alcohol',
                'consumption_amphetamines', 
                'consumption_cannabis',
                'consumption_mushrooms', 'consumption_nicotine']] == period).sum(axis=1)})

    return df


# In[23]:


drugs_df = Feature(drugs_df)


# In[24]:


drugs_df.head()


# ## Data Cleaning

# ### Key takeaways:
# 
# 1. The "ID" variable has no use, so we drop it
# 2. "consumption_cocaine_last_month" variable is Binary (Yes/No), for SKLERN we need to label them 1 and 0, respectively.
# 3. Age, Consumption, Education variable are not ordered, we provide an order as ordinals.

# In[25]:


age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']

education_order = ['Left school before 16 years',
                   'Left school at 16 years',
                   'Left school at 17 years',
                   'Left school at 18 years',
                   'Some college or university, no certificate or degree',
                   'Professional certificate/ diploma',
                   'University degree',
                   'Masters degree',
                   'Doctorate degree']

consumption_order = ['never used',
                     'used over a decade ago',
                     'used in last decade',
                     'used in last year',
                     'used in last month',
                     'used in last week',
                     'used in last day']

consumption_columns = ['consumption_alcohol',
                       'consumption_amphetamines', 'consumption_caffeine',
                       'consumption_cannabis', 'consumption_chocolate',
                       'consumption_mushrooms', 'consumption_nicotine']

def clean_df(df: pd.DataFrame, train_data:bool=True) -> pd.DataFrame:
    df = df.drop('id', axis=1)
    if train_data:
        df['consumption_cocaine_last_month'] = df['consumption_cocaine_last_month'].replace({'No': 0, 'Yes': 1})
    df['age'] = df['age'].apply(lambda e: age_order.index(e))
    df['education'] = df['education'].apply(lambda e: education_order.index(e))

    for consumption_column in consumption_columns:
        df[consumption_column] = df[consumption_column].apply(lambda e: consumption_order.index(e))
    
    return df


# In[26]:


drugs_df = clean_df(drugs_df)


# In[27]:


drugs_df.head()


# ## General Notes

# 1. The ideal split is said to be 80:20 for training and testing. In our data, the observation count is not high, so it will be easier to have a good results in such case. That's why we continue with 70:30 split.
# 2. Random split each time to check consistency of results for each model. 
# 3. Preprocess the data: categorical feature encoding, numeric feature scaling
# 4. OneHotEncoder: Encode categorical integer features using a one-hot aka one-of-K scheme.
# 5. Fine-tune hyperparameters using grid search with 3-fold cross-validation. Due to limited datasize, we go with 3-fold instead of 5.
# 6. Model is retrained on the whole training set using the best hyperparameters obtained using grid search.
# 7. Checking overfitting by comparing "Balanced accuracy" of training and test sets.
# 
# #### The best model is chosen based on the highest "Balanced Accuracy".
# 

# ## Dataset splitting

# In[28]:


cat_features = ['gender', 'country', 'ethnicity']

num_features = ['personality_neuroticism', 'personality_extraversion',
                 'personality_openness', 'personality_agreeableness',
                 'personality_conscientiousness', 'personality_impulsiveness',
                 'personality_sensation', 'no_illicit_drugs', 'TAD_last_day', 'TAD_last_week',
                 'TAD_last_month', 'TAD_last_year',
                 'TAD_last_decade', 'TAD_decade_ago']

ordinal_features = ['age', 'education', 'consumption_alcohol',
                    'consumption_amphetamines', 'consumption_caffeine',
                    'consumption_cannabis', 'consumption_chocolate',
                    'consumption_mushrooms', 'consumption_nicotine']

target = 'consumption_cocaine_last_month'


# For ordered features we'll manually provide the order of levels to be used in transformations:

# In[29]:


X = drugs_df[cat_features + num_features + ordinal_features]
y = drugs_df[target]


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.7)


# ## Benchmark

# Dummy Classifier prediction has a 0.5 Balanced Accuracy Score. Our intention is to increase the Accuracy, higher is the better.

# In[31]:


dummy_classifier = DummyClassifier()
dummy_classifier.fit(X_train, y_train)

balanced_accuracy_score(y_train, dummy_classifier.predict(X_train))


# ## Logistic regression

# Logistic regression is a classification algorithm, used when the value of the target variable is categorical in nature. Logistic regression is most commonly used when the data in question has binary output, so when it belongs to one class or another, or is either a 0 or 1.
# 
# Regularization is a way to avoid overfitting by penalizing high-valued regression coefficients:
# Regularization works by biasing data towards particular values (such as small values near zero). The bias is achieved by adding a tuning parameter to encourage those values:
# 1. L1 regularization adds an L1 penalty equal to the absolute value of the magnitude of coefficients. In other words, it limits the size of the coefficients.
# 2. L2 regularization adds an L2 penalty equal to the square of the magnitude of coefficients. L2 will not yield sparse models and all coefficients are shrunk by the same factor.
# 3. C Hyperparameter to control the model how to choose parameters.

# In[32]:


ohe_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

scaler = MinMaxScaler()
transformers = ColumnTransformer([('categorical', ohe_transformer, cat_features),
                                  ('numeric', scaler, num_features)], remainder='passthrough')
logreg_pipeline = Pipeline(steps=[('transform', transformers),
                                  ('logreg', LogisticRegression(solver='liblinear', class_weight='balanced'))])


# In[33]:


logreg_param_grid = {'logreg__penalty' : ['l1', 'l2'],
                     'logreg__C' : np.logspace(-4, 4, 20)}


# In[34]:


log_search = GridSearchCV(logreg_pipeline, logreg_param_grid, scoring='balanced_accuracy', cv=3)


# In[35]:


log_search.fit(X_train, y_train)


# In[36]:


log_search.best_estimator_


# ### Classification Report Guide (General Explanation)
# 1. Precision — What percent of predictions were correct?
# 2. Recall — What percent of the positive cases caught?
# 3. F1 score — What percent of positive predictions were correct?

# In[37]:


Results(y_train, log_search.predict(X_train))


# In[38]:


Results(y_test, (log_search.predict(X_test)))


# ## SVM

# Support Vector Machines are one of the most popular and widely used algorithm for dealing with classification problems in machine learning.
# 
# Considered Hyperparameters:
# 1. Kernels: The main function of the kernel is to take low dimensional input space and transform it into a higher-dimensional space. It is mostly useful in non-linear separation problem.
# 2. Gamma: It defines how far influences the calculation of plausible line of separation. (higher Gamma, nearby points will have high influence)
# 3. C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimisation how much error is bearable. This is how to control the trade-off between decision boundary and misclassification term.
# 

# In[39]:


ohe_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

scaler = MinMaxScaler()
transformers = ColumnTransformer([('categorical', ohe_transformer, cat_features),
                                  ('numeric', scaler, num_features)], remainder='passthrough')
svm_pipeline = Pipeline(steps=[('transform', transformers),
                                  ('svm', SVC(class_weight='balanced'))])


# In[40]:


svm_param_grid = {'svm__C': [0.1,1, 10, 100], 'svm__gamma': [1,0.1,0.01,0.001],'svm__kernel': ['rbf', 'poly', 'sigmoid']}


# In[41]:


svm_search = GridSearchCV(svm_pipeline, svm_param_grid, scoring='balanced_accuracy', cv=3)


# In[42]:


svm_search.fit(X_train, y_train)


# In[43]:


svm_search.best_estimator_


# In[44]:


Results(y_train, svm_search.predict(X_train))


# In[45]:


Results(y_test, svm_search.predict(X_test))


# ## KNN

# ### K Nearest Neighbor Classification Algorithm
# 
# It is one of the simplest and widely used classification algorithms in which a new data point is classified based on similarity in the specific group of neighboring K data points.
# 
# Hyperparameters:
# 1. K-neighbors - The number of the neighbors
# 2. knn_p -> Minkowski distance p; the Manhattan distance, p=1 and the Euclidean, p=2 .

# In[46]:


ohe_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
ordinal_transformer = OrdinalEncoder(categories=[age_order, education_order] + [consumption_order] * 7)
scaler = MinMaxScaler()
transformers = ColumnTransformer([('categorical', ohe_transformer, cat_features),
                                  ('numeric', scaler, num_features)], remainder='passthrough')
knn_pipeline = Pipeline(steps=[('transform', transformers),
                                  ('knn', KNeighborsClassifier())])


# In[47]:


knn_param_grid = {'knn__n_neighbors': np.arange(5, 31),
                  'knn__p': [1, 2]}


# In[48]:


knn_search = GridSearchCV(knn_pipeline, knn_param_grid, scoring='balanced_accuracy', cv=3)


# In[49]:


knn_search.fit(X_train, y_train)


# In[50]:


knn_search.best_estimator_


# In[51]:


Results(y_train, knn_search.predict(X_train))


# In[52]:


Results(y_test, knn_search.predict(X_test))


# ## Applying the best model on the test data

# The best Balanced Accuracy Score is given by Logistic Regression. So we will use that model to predict on the TEST data.
# 
# ###### Balanced Accuracy Score: 0.7965836149142396 (Train)
# ###### Balanced Accuracy Score: 0.7371614716402657 (Test)

# In[53]:


logreg_best_params = log_search.best_params_


# In[54]:


ohe_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
scaler = MinMaxScaler()
transformers = ColumnTransformer([('categorical', ohe_transformer, cat_features),
                                  ('numeric', scaler, num_features)], remainder='passthrough')
best_logreg_pipeline = Pipeline(steps=[('transform', transformers),
                                  ('logreg', LogisticRegression(solver='liblinear',
                                                                class_weight='balanced',
                                                                C=logreg_best_params['logreg__C'],
                                                                penalty=logreg_best_params['logreg__penalty']))])


# In[55]:


best_logreg_pipeline.fit(X, y)


# In[56]:


drugs_test_df = pd.read_csv('drugs_test.csv')


# In[57]:


output_df = pd.DataFrame()
output_df['id'] = drugs_test_df['id']


# In[58]:


drugs_test_df.head()


# No missing values in Test data

# In[59]:


drugs_test_df.info()


# In[60]:


drugs_test_df = Feature(drugs_test_df)
drugs_test_df = clean_df(drugs_test_df, train_data=False)


# In[61]:


drugs_test_df


# In[62]:


output_df['consumption_cocaine_last_month'] = pd.Series(best_logreg_pipeline.predict(drugs_test_df)).replace({0: 'No', 1: 'Yes'})


# In[63]:


output_df


# In[64]:


output_df.to_csv('classification.csv', index=False)

