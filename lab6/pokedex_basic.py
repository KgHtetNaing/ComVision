import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


print ("hello pokedex")
# print(np.__version__)
# print(pd.__version__)


dat = pd.read_csv("pokemon.csv")

dat['sum_stat'] = dat['attack'] + dat['defense'] + dat['special_attack'] + dat['special_defense'] + dat['speed'] + dat['hp']

def categorize_strength(val):
    if val <= 300:
        return 0    #weak
    elif val <= 350:
        return 1    #mid
    else:
        return 2    #strong
    
def categorize_height(val):
    if val <= 9: return 0   #small
    elif val < 15:return 1  #mid
    else : return 2         #big



dat['strength_label'] = dat['sum_stat'].apply(categorize_strength)
dat['height_label']  = dat['height'].apply(categorize_height)

# print(dat['strength_label'].head(10))
# print(dat['height_label'].head(10))

Y = dat[['strength_label']]
X = dat[['height_label']]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size=0.2)
print (len(X_train))
print (len(X_test))

model = DecisionTreeClassifier()
model.fit(X_train , Y_train) #put all the data into model to learn

Y_pred = model.predict(X_test)

print(accuracy_score(Y_test , Y_pred))




#print (dat) -- all data
#print (dat[column_name]) -- give that column
#print (dat[name] == value) -- Rows that are true
#print (dat [dat (name) == value]) - rows with said value
#print (dat [dat (name) == value] [[name1, name2]])  