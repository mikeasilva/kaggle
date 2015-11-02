import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import SelectKBest, f_classif

number_of_predictors = 4

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('data/train.csv', header=0)

df.describe()
gender_map = {'female':1, 'male':0}
df['Gender'] = df['Sex'].map(gender_map).astype(int)

# Step 1 = Fill in the missing embarked with the mode
embarked_counts = df.groupby('Embarked').size()
print('Counts Before')
print(embarked_counts)
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)
embarked_counts = df.groupby('Embarked').size()
print("\n"+'Counts After')
print(embarked_counts)
embarked_map = {'C':0, 'Q':1, 'S':2} 
df['EmbarkedInt'] = df['Embarked'].map(embarked_map).astype(int)

# Step 2 = Fill in the missing age
# I want to fill in the missing age variables using the median age of the
# people with the same title

def get_title(name):
    split_name = name.split(',')
    split_name = split_name[1].split('.')
    title = split_name[0].strip()
    return (title)

# First I need to get the title broken out of the name  
df['Title'] = df['Name'].apply(lambda x: get_title(x))

# Next I need to get the median age by title
median_age_by_title = df.groupby('Title')['Age'].agg([np.median])['median'].to_dict()

# Finally I can fill in the missing ages
for index, row in df.iterrows():
    if np.isnan(row['Age']):
        df.at[index, 'Age'] = median_age_by_title[row['Title']]

# Step 3 = Map the titles to integers
# There are some titles that are not numerious so they are collapsed.  French
# versions are grouped up with their english equivalents
title_counts = df.groupby('Title').size()
print("\n"+'Title Counts')
print(title_counts)
title_map = {'Capt': 1, 
             'Col': 2, 'Major': 2,
             'Don': 3, 'Sir': 3,
             'Dr': 4, 
             'Dona':5, 'Jonkheer': 5, 'Lady': 5, 'the Countess':5,            
             'Master': 6,
             'Miss': 7, 'Mlle': 7, 'Ms': 7,
             'Mme': 8, 'Mrs': 8,
             'Mr': 9,
             'Rev': 10}
df['TitleInt'] = df['Title'].map(title_map).astype(int)

predictors = ["Pclass", "Gender", "Age", "SibSp", "Parch", "Fare", "EmbarkedInt", 'TitleInt']

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(df[predictors], df["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
predictors_df = pd.DataFrame({'Predictor':predictors, 'P Value':scores}).sort('P Value', ascending=0).head(number_of_predictors)

predictors = predictors_df['Predictor'].tolist()

y_train = pd.Series(df['Survived'])
x_train = df[predictors]

# Create the random forest object
rf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(rf, x_train, y_train, cv=5)
print(np.mean(scores))

# Fit the training data to the Survived labels and create the decision trees
forest = rf.fit(x_train, y_train)

# Take the same decision trees and run it on the test data
test_data = pd.read_csv('data/test.csv', header=0)
test_data.describe()

# Transform gender
test_data['Gender'] = test_data['Sex'].map(gender_map).astype(int)

# Fill in Title
test_data['Title'] = test_data['Name'].apply(lambda x: get_title(x))        
test_data['TitleInt'] = test_data['Title'].map(title_map).astype(int)
   
# Fill in the missing ages
for index, row in test_data.iterrows():
    if np.isnan(row['Age']):
        test_data.at[index, 'Age'] = median_age_by_title[row['Title']]

# Fill in missing embaked
test_data['Embarked'] = test_data['Embarked'].fillna(embarked_mode)
test_data['EmbarkedInt'] = test_data['Embarked'].map(embarked_map).astype(int)

# Fill in missing fare
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

PassengerId = pd.Series(test_data['PassengerId'])

output = rf.predict(test_data[predictors])

prediction = {'PassengerId':PassengerId, 'Survived':output}
results = pd.DataFrame(prediction)
results.to_csv('python.csv', index = False)