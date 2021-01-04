
import pandas as pd
import numpy as np

import random
import copy

from functions import (
        Node,
        make_columns_numerical,
        modify_fringe_cases,
        calculate_impurity,
        split_dataset,
        calculate_best_split, 
        tree_prediction,
        test_accuracy,
        prune,
        forest_prediction,
        forest_accuracy,
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

#____________________________
        
# Growing the unpruned tree
def grow_unpruned_tree(df, var_list, output_var):   
    root_node = Node(calculate_impurity(df[output_var]))
  
    current_node = root_node
    current_df = df
      
    stack = []
    
    cond = True
    while cond:  
        # For the current node...
        # Find the best variable & value with which to split the dataset
        best_split = calculate_best_split(current_df, var_list, output_var)  
        df1, df2 = split_dataset(current_df, best_split)
        left_impurity = calculate_impurity(df1[output_var])
        right_impurity = calculate_impurity(df2[output_var])    
        # Create left & right nodes for the current node
        left_node = Node(left_impurity)
        right_node = Node(right_impurity)
        
        current_node.do_split(left_node, right_node, best_split)
        
        # If the data subsets are pure, we have a leaf node.
        # Get a node from stack (explained in the next comment) to expand next
        if left_impurity == right_impurity == 0:
            if len(stack) == 0:
                cond = False
            else:
                current_node = stack[-1][0]
                current_df = stack[-1][1]
                del stack[-1]
        
        # Choose the more impure region for the next node to expand. 
        # The other one goes on the stack.           
        elif left_impurity > right_impurity:
            current_node = left_node
            current_df = df1
            if right_impurity != 0:
                stack.append([right_node, df2])
        else:
            current_node = right_node
            current_df = df2
            if left_impurity != 0:
                stack.append([left_node, df1])
                
    return root_node
             
    
#____________________________
    
# Perform the pruning procedure on copies of the tree for various values of ...
# Alpha for the cost function. Return the tree copy with the best performance
def grow_decision_tree(df_train, df_test, var_list, output_var):
    trees, accuracies = [], []
    alpha = 0 
    
    root_node_unpruned = grow_unpruned_tree(df_train, var_list, output_var)
       
    cond = True
    while cond:
        root_node = copy.deepcopy(root_node_unpruned)
        prune(root_node, df_train, var_list, output_var, df_test, root_node, alpha)
        acc = test_accuracy(root_node, df_train, var_list, output_var, df_test)
        trees.append(root_node)
        accuracies.append(acc)
        if root_node.left == None and root_node.right == None:
            break
        alpha += 0.02
    
    final_model = trees[accuracies.index(max(accuracies))]
    
    return final_model

#____________________________

# Grow the forest

forest = []

num_trees = 10
for i in range(num_trees):
    randoms = np.array(random.sample(range(len(df)), len(df)))
    mask = randoms < int(0.8*len(df))
    
    df_train = df[mask]
    df_test = df[~mask]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    print("Growing tree number " + str(i))
    tree = grow_decision_tree(df_train, df_test, var_list, output_var)
    forest.append([tree, df_train])


# Final test
    
# Make a new random set of test data and check the forest's performance on it
randoms = np.array(random.sample(range(len(df)), len(df)))
mask = randoms < int(0.8*len(df))
df_test = df[~mask]

print("Accuracy of random forest:")
print(forest_accuracy(forest, df_test, var_list, output_var))










