# Credit Risk Analysis Overview
# Choosing a dataset

I probably spent the most amount of time on this assignment choosing a dataset. I was trying to find something binary that I could classify, specifically something that has nulls or redundant columns or whatever, but I couldn't really find anything I liked. The dataset I did choose, "Credit Card Approval", ended up being mostly clean -- no nulls, *mostly* relevant columns, etc. 
## The Dataset itself // Cleaning the data

The dataset is pretty self explanatory. Each entry is a credit card applicant. The features that describe each applicant include categorical data like their age, gender, # of children, marital status, whether or not they own property or a car, etc. It also includes their yearly income. The target variable (aptly named `'TARGET'`) is already binarized -- 1 for credit risks, 0 for non-risks. 

Despite the data being relatively clean I did note that for classification purposes there were some redundant features. Of which, the most self explanatory would be `'ID', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'`. Not sure exactly why we would need to know if applicants have phones or emails. `ID` is also a non-descriptive column in terms of prediction. 

I also removed `'STATUS'`, which to me is still unclear (values were 0, X, C, ...?) and `'NAME_HOUSING_TYPE'`, which I found redundant given we had a flag for whether or not applicants owned property. 

I also performed some small conversions on the numerical data as some of them were stored as negative values. An example of this is `'DAYS_EMPLOYED`, where the values were negative. I'm not exactly sure why this was the case, but my assumption is that the calculation used for these columns subtracted certain dates (like `'employment_date'`) from the date the application was submitted. 
## Outliers Removed

I don't believe this needs much explanation. I used the $\text{IQR}$ formula to find outliers based on applicant total income, total number of days employed, and age. These are outliers I feel could skew the classification results later down the line. 
## MinMax Normalization

Some classification models, like KNN (which is distance-based) may be sensitive to large numbers like salary. I felt that if I was going to use KNN as a classification model it was necessary to scale the salary (and I did `'DAYS_EMPLOYED'` as well) down to a much smaller range so there wouldn't be any issues. 
# Sampling the Data

Another ***massive*** issue with this dataset is the extreme class imbalance in the `'TARGET'` feature. 

`df_cln['TARGET'].value_counts()` nets:

| TARGET |        |
| ------ | ------ |
| 0      | 481264 |
| 1      | 1730   |

Funny thing -- I tried to ignore this at first because I have not done any sampling before. Then, when I actually trained the DecisionTree model, the training and test scores were both $\approx 0.99$. So either, the DT model was really excellent, or it was more likely that the model was overfitting. So, I gave up and decided to try my hand at sampling. 

I'm not exactly well-versed in the sampling techniques so the only thing that made sense to me was to sample the data so that the target column was 50-50. I'm sure that there is a better way to do this, and frankly this might even violate the requirement of having 10,000+ rows for this assignment, but I wasn't exactly sure how else to do this without getting too complicated. So, I split the data by target values 0 and 1, and used the sample function to take $n$ samples of the majority class (where $n$ is the size of the minority class), and strung the dataset back together through concatenation. This netted me:

| TARGET |      |
| ------ | ---- |
| 1      | 1730 |
| 0      | 1730 |

Again, I didn't have any other methods. This results in what I would describe as "catastrophic" data loss, but it makes for a more reliable model. So maybe I'll learn more about sampling methods down the line. 
# Models
## KNN

I tested $k$ from $2$ to $20$. These were the results:
```
K: 2 | Training Score: 0.9436416184971098 | Test Score: 0.7789017341040463
K: 3 | Training Score: 0.8659682080924855 | Test Score: 0.7210982658959537
K: 4 | Training Score: 0.8638005780346821 | Test Score: 0.7326589595375722
K: 5 | Training Score: 0.8171965317919075 | Test Score: 0.6950867052023122
K: 6 | Training Score: 0.8276734104046243 | Test Score: 0.6950867052023122
K: 7 | Training Score: 0.7933526011560693 | Test Score: 0.6589595375722543
K: 8 | Training Score: 0.7742052023121387 | Test Score: 0.6777456647398844
K: 9 | Training Score: 0.7554190751445087 | Test Score: 0.6445086705202312
K: 10 | Training Score: 0.7539739884393064 | Test Score: 0.6546242774566474
K: 11 | Training Score: 0.7463872832369942 | Test Score: 0.634393063583815
K: 12 | Training Score: 0.7453034682080925 | Test Score: 0.6459537572254336
K: 13 | Training Score: 0.7348265895953757 | Test Score: 0.6315028901734104
K: 14 | Training Score: 0.7290462427745664 | Test Score: 0.638728323699422
K: 15 | Training Score: 0.7210982658959537 | Test Score: 0.630057803468208
K: 16 | Training Score: 0.7174855491329479 | Test Score: 0.634393063583815
K: 17 | Training Score: 0.7048410404624278 | Test Score: 0.6098265895953757
K: 18 | Training Score: 0.6997832369942196 | Test Score: 0.619942196531792
K: 19 | Training Score: 0.7005057803468208 | Test Score: 0.6141618497109826
K: 20 | Training Score: 0.6929190751445087 | Test Score: 0.630057803468208
```

Honestly, not the greatest. The best scores are from KNN models (2, 3) that arguably have yet to see enough diverse data. So, I chose the best KNN to be $k=6$. It has the best training and testing score without being severely undertrained, or over/underfit. 
## DecisionTrees

Using the DecisionTreeClassifier we netted scores:
```
Train: 0.9978323699421965
Test: 0.884393063583815
```

DTC did suspiciously well on the training data. But then also did not-so-suspiciously well on the test data.

To find the best DecisionTree model, I used `GridSearchCV` to find the best model with the best F1 score. 
## Naive Bayes

Naive Bayes did extremely poorly on this dataset. Not sure why.
```
Train: 0.5924855491329479
Test: 0.5606936416184971
```
# Model Evaluation

I followed the playbook for PA2 and scored each model based on F1, precision, and recall. Note the results:
```
--------------+------------------+-----------+--------+----------+
|    model     | confusion matrix | precision | recall | F1 score |
+--------------+------------------+-----------+--------+----------+
|     KNN      |    [[226 127]    |   0.6675  | 0.7522 |  0.7074  |
|              |    [ 84 255]]    |           |        |          |
|              |                  |           |        |          |
| DecisionTree |    [[276  77]    |   0.7902  | 0.8555 |  0.8215  |
|              |    [ 49 290]]    |           |        |          |
|              |                  |           |        |          |
|  NaiveBayes  |    [[215 138]    |   0.5563  | 0.5103 |  0.5323  |
|              |    [166 173]]    |           |        |          |
+--------------+------------------+-----------+--------+----------+
```

Here it's important to consider the best metric for choosing the best model. Since the data is credit-risk-based, where we're predicting whether or not an applicant is a credit risk, it's important to give weight to the number of false negatives. 

Based upon the scores, it's clear that the best model for this data is the DecisionTree. It has the lowest number of false negatives, and the best scores by a pretty wide margin compared to the other models. 
