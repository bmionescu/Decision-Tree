
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


from functions import (
        make_columns_numerical,
        modify_fringe_cases,
)
    
#____________________________

# Data 

df = pd.read_excel('./meal_data.xlsx') # df for data frame
# These are all linearly dependent on 'EER[kcal]'
df.drop(['P target(15%)[g]', 'F target(25%)[g]', 'C target(60%)[g]'], axis=1, inplace=True)

df_keys = df.keys()
var_list = list(df_keys[0:-1]) # Independent variables
output_var = df_keys[-1] # Output variable

# Some columns have string data. Replace them with categorical integers    
df = make_columns_numerical(df, df_keys)  
  
# If a row of data is identical except for the output variable ....
# Modify the Vegetable[g] entry slightly, to break the mathematical fringe case
modify_fringe_cases(df, var_list)

print("Finished loading and cleaning data\n")

X = df.iloc[:, 0:-1] # Input data
y = df.iloc[:, -1] # Output data

#____________________________

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

random_forest = RandomForestClassifier()
model = random_forest.fit(X_train, y_train)

y_test_predictions = model.predict(X_test)

print(accuracy_score(y_test, y_test_predictions))

count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] == y_test_predictions[i]:
        count += 1






