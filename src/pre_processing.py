from input import *

# dropping columns that are not important for the model.
df.drop(['year', 'team'], axis = 1, inplace = True)
#print(df.champions.value_counts())

# new features
df['win_match_ratio'] = (df['wins'] + 1)/ df['match_played']
df['gs_match_ratio'] = (df['goals_scored'] + 1)/ df['match_played']
df['gc_match_ratio'] = (df['goals_conceded'] + 1)/ df['match_played']
df['win_gs_ratio'] = (df['wins'] + 1)/(df['goals_scored'] + 1)
df['win_lost_ratio'] = (df['wins'] + 1)/(df['losts'] + 1)
df['gs_gc'] = (df['goals_scored'] - df['goals_conceded']) + 0.1
df['wins_draws_ratio'] = (df['wins'] +  1) / (df['draws'] + 1)
df['gs_gd'] = (df['goals_scored'] + 1) + (df['gd'])

#print(df.win_lost_ratio.describe())

#finding important features
correlation = df.corr()
correlation.sort_values(["champions"], ascending = False, inplace = True)
print(correlation.champions)

# independent and dependent matrix of feature
x = df.drop('champions', axis = 1)
y = df['champions']

# balancing the dataset
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_sm, y_sm= smote.fit_resample(x, y)

# splitting into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_sm, y_sm,random_state = 31)

