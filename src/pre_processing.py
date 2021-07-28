from input import *

# dropping columns that are not important for the model.
df.drop(['year', 'team'], axis = 1, inplace = True)

# train and dev set
train = df.loc[:550]
dev = df.loc[551:]

#print(train.champions.value_counts())
#print(dev.champions.value_counts())

x = train.drop('champions', axis = 1)
y = train['champions']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 31)


