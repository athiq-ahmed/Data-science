# https://datascienceplus.com/multi-class-text-classification-with-scikit-learn/
# http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/
# https://gist.github.com/yusugomori/4462221
# https://github.com/susanli2016/Machine-Learning-with-Python
# http://blog.yhat.com/posts/predicting-customer-churn-with-sklearn.html


import pandas as pd
import tabulate as tb
path = r'C:\Users\athiq.ahmed\Desktop\Other\Python code\Datasets\Consumer_Complaints.csv'
df = pd.read_csv(path)


categorical = df.dtypes[df.dtypes=='object'].index
categorical

Numerical = df.dtypes[df.dtypes!='object'].index
Numerical

print("Total number of observations are", df.shape[0])  ## 1,076,212
print("Total number of columns are %s. The categorical variables are %s and Numerical variables are %s "
      %(df.shape[1],(len(categorical)),(len(Numerical))))
print(tb.tabulate(df.head(), headers='keys',tablefmt='psql'))

# from io import StringIO
col = ['Product', 'Consumer complaint narrative']
df =df[col]
df.shape
df.isnull().sum()/df.isnull().count() *100
df = df[pd.notnull(df['Consumer complaint narrative'])]
df.shape  ## (301,348, 2)

df.isnull().sum()
df['category_id'] = df['Product'].factorize()[0]
print(tb.tabulate(df.head(),headers='keys',tablefmt='psql'))
len(df.Product.unique())
category_id_df = df[['Product','category_id']].drop_duplicates().sort_values('category_id')
category_id_df.head()
category_to_id = dict(category_id_df.values)
category_to_id
id_to_category = dict(category_id_df[['category_id','Product']].values)
id_to_category
df.head()
print(tb.tabulate(df.head(),headers='keys',tablefmt='psql'))

df.rename(columns={'Consumer complaint narrative':'Consumer_complaint_narrative'},inplace=True)
df.head()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Product').Consumer_complaint_narrative.count().plot.barh()
plt.show()

df.Product.value_counts(sort=True)
len(df.Product.unique())

print(tb.tabulate(df.head(), headers='keys',tablefmt='psql'))

# Using .isin  -- selecting or slicing the data
product_lists = ['Debt collection','Mortgage','Credit reporting','Credit card','Student loan','Bank account or service','Consumer Loan']
df_new = df[df['Product'].isin(product_lists)]
df_new.head()
df_new.shape
print(tb.tabulate(df.head(), headers='keys',tablefmt='psql'))
len(df_new.Product.value_counts(sort=True))
df_new['Product'].value_counts()

# Text reprsentation
"""
* sublinear_df is set to True to use a logarithmic form for frequency.
* min_df is the minimum numbers of documents a word must be present in to be kept.
* norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1.
* ngram_range is set to (1, 2) to indicate that we want to consider both unigrams and bigrams.
* stop_words is set to “english” to remove all common pronouns (“a”, “the”, …) to reduce the number of noisy features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Consumer_complaint_narrative)
labels = df.category_id
features.shape


""""
We can use sklearn.feature_selection.chi2 to find the terms that are the most correlated with each of the products
"""

from sklearn.feature_selection import chi2
import numpy as np
N = 2
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


# Multi-Class Classifier: Features and Design

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df_new['Consumer_complaint_narrative'], df_new['Product'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# After fitting the training set, let’s make some predictions.
print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
df[df['Consumer_complaint_narrative'] == "This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."]

print(clf.predict(count_vect.transform(["I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"])))
df[df['Consumer_complaint_narrative'] == "I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"]


# Model selection
