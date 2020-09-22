# https://elitedatascience.com/imbalanced-classes
# https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation

            # Up-sample the minority class
            # Down-sample the majority class
            # Change your performance metric
            # Penalize algorithms (cost-sensitive training)
            # Use tree-based algorithms

# Reason to upsample or downsample
    # The main motivation behind the need to preprocess imbalanced data before we feed them into a classifier is that typically classifiers
    # are more sensitive to detecting the majority class and less sensitive to the minority class. Thus, if we don't take care of the issue,
    # the classification output will be biased, in many cases resulting in always predicting the majority class


import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv(r'C:\Users\athiq.ahmed\Desktop\Other\Python code\Imbalanced classes wip\Dataset\balance-scale.csv',
                 names = ['balance', 'var1','var2','var3','var4'])

# display the observations
df.head()

# count for each class
df['balance'].value_counts()

# Transform into binary class
df['balance_new'] = [1 if i=='B' else 0 for i in df.balance]
df.head()
df.balance_new.value_counts()/df.balance_new.count()*100

# The danger of imbalanced classes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Fit the model using default settings
# separate input features(X) and target variable(y)
df.head()
y = df.balance_new;y.head()
X = df.drop(['balance', 'balance_new'], axis=1);X.head()

# Train the model
clf_0 = LogisticRegression().fit(X,y);clf_0

# predict on the training set
pred_y_0 = clf_0.predict(X);

# How's the accuracy
accuracy_score(pred_y_0,y)

# Is it predicting only class 1
np.unique(pred_y_0)  # this model is predicting only 0, which means it is completely ignoring the minority class in favor of the majority class


# Techniques for handling imbalanced-class
# 1. up-sampling the minority class  -- Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.
# resample with replacement
        # First, we'll separate observations from each class into different DataFrames.
        # Next, we'll resample the minority class with replacement, setting the number of samples to match that of the majority class.
        # Finally, we'll combine the up-sampled minority class DataFrame with the original majority class DataFrame.


from sklearn.utils import resample

# steps:
# 1. separate majority and minority classes
df_majority = df[df.balance_new==0]
df_minority = df[df.balance_new==1]

len(df_majority)
len(df_minority)

# 2. upsample minority class -- we'll resample the minority class with replacement, setting the number of samples to match that of the majority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,       # sample with replacement
                                 n_samples=576,      # to match majority class
                                 random_state=123)   # reproducible results

df_minority_upsampled.head()

# 3. combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled]);df_upsample.head()

# Display new class counts
df_upsampled.balance_new.value_counts()


# Train the model on upsampled dataset
# separate input features and target variables
y= df_upsampled.balance_new;y.head()
X= df_upsampled.drop(['balance','balance_new'],axis=1);X.head()

# Train the model
clf_1 = LogisticRegression().fit(X,y)

# predict on the training set
pred_y_1 = clf_1.predict(X)

# Is our model still predicting class 1 ?
np.unique(pred_y_1)

# how's the accuracy?
accuracy_score(y,pred_y_1)


# Down sample Majority class  -- Down-sampling involves randomly removing observations from the majority class to prevent its signal from dominating the learning algorithm
# resampling without replacement
        # First, we'll separate observations from each class into different DataFrames.
        # Next, we'll resample the majority class without replacement, setting the number of samples to match that of the minority class.
        # Finally, we'll combine the down-sampled majority class DataFrame with the original minority class DataFrame

#1.Separate majority and minority classes
df_majority = df[df.balance_new==0]
df_minority = df[df.balance_new==1]

#2. Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,       # Sample without replacement
                                   n_samples=49,        # to match minority class
                                   random_state=123)    # reproducible results
df_majority_downsampled.head()

#3. Combine majority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled.head()

# Display the new class counts
df_downsampled.balance_new.value_counts()

# Again,  let's train our model
# Separate input features and target variable
y = df_downsampled.balance_new
X= df_downsampled.drop(['balance','balance_new'], axis=1)

# train model
clf_2 = LogisticRegression().fit(X,y)

# predict the model
pred_y_2 = clf_2.predict(X)

# Is our model still predicting just one class?
np.unique(pred_y_2)

# Accuracy
accuracy_score(y,pred_y_2)


#3.Change your performance metric
    # AUROC - AUROC represents the likelihood of your model distinguishing observations from two classes
    # In other words, if you randomly select one observation from each class, what's the probability that
        # your model will be able to "rank" them correctly?

from sklearn.metrics import roc_auc_score

    # To calculate AUROC, you'll need predicted class probabilities instead of just the predicted classes.
    # You can get them using the .predict_proba()  function like so:

# preict class probabilities
prob_y_2 = clf_2.predict_proba(X)
prob_y_2[:5]

# Keep only the positive class
prob_y_2 =[i[1] for i in prob_y_2]
prob_y_2[:5]

# AUROC of model trained on downsampled dataset
roc_auc_score(y,prob_y_2)

# AUROC of model trained on imbalanced dataset
prob_y_0 = clf_0.predict_proba(X)
prob_y_0 = [i[1] for i in prob_y_0]

roc_auc_score(y,pred_y_0)


#4. Penalize algorithms
    # The next tactic is to use penalized learning algorithms that increase the cost of classification mistakes on the minority class


from sklearn.svm import SVC

# Train penalized svm on imbalanced dataset
# separate input features (X) and target variable(y)
y = df.balance_new
X= df.drop(['balance', 'balance_new'], axis =1)

# Train model
clf_3 = SVC(kernel='linear',
            class_weight='balanced',  # penalize
            probability=True)
clf_3.fit(X,y)

# predict on training set
pred_y_3 = clf_3.predict(X)

# Is our model still predicting just one class ?
np.unique(pred_y_3)

# accuracy
accuracy_score(y, pred_y_3)

# AUROC
prob_y_3 = clf_3.predict_proba(X)
prob_y_3 = [i[1] for i in prob_y_3]
roc_auc_score(y,prob_y_3)


#5. Use tree based algorithms
from sklearn.ensemble import RandomForestClassifier

# train random forest on imbalanced dataset
# Separate input features(X) and target variable(y)
y = df.balance_new
X = df.drop(['balance', 'balance_new'], axis =1)

# train model
clf_4= RandomForestClassifier()
clf_4.fit(X,y)

# predict model
pred_y_4 = clf_4.predict(X)

# Is our model still predicting just one class ?
np.unique(pred_y_4)

# Accuracy
accuracy_score(y, pred_y_4)

# AUROC
prob_y_4 = clf_4.predict_proba(X)
prob_y_4 = [i[1] for i in prob_y_4]
roc_auc_score(y, prob_y_4)


