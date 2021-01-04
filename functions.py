
# Functions for the Decision tree

import numpy as np
from collections import Counter
import math

#____________________________

# Defining the node objects

class Node():
    def __init__(self, purity):
        self.left = None # Object
        self.right = None # Object
        self.split = None # List [axis, value]
        self.purity = purity # scalar
               
    def do_split(self, left, right, split):
        self.left = left
        self.right = right
        self.split = split
        
#

# Define which string entries correspond to which numerical values 
def convert_string(string):
    if string == 'breakfast':
        return 1
    if string == 'lunch':
        return 2
    if string == 'dinner':
        return 3
    if string == 'female':
        return 1
    if string == 'male':
        return 2
    
#

# Convert dataset's string entries to integers representing their categories  
# Also delete rows that are missing an entry
def make_columns_numerical(df, df_keys):
    for i in range(len(df)):
        nan_row = False
        for j in df_keys:
            if type(df[j][i]) == str:
                df.loc[i, j] = convert_string(df[j][i])
            elif math.isnan(df[j][i]):
                nan_row = True
        if nan_row: 
            df = df.drop(i)

    df = df.reset_index(drop=True) 
    
    return df
        
#
       
# Account for fringe cases where two rows have same input values, different outputs 
def modify_fringe_cases(df, var_list):
    concat_entries = []
    for j in range(len(df)):
        concat_string = ''
        for i in var_list:
            concat_string += str(df[i][j])
        concat_entries.append(concat_string)
     
    concat_entries = np.array(concat_entries)
    repeated_entries = Counter(concat_entries)
    for key, elem in repeated_entries.items():
        if elem != 1:
            locations = np.where(concat_entries == key)
            for count, i in enumerate(locations[0]):
                df.loc[i, 'Vegetables[g]'] = df.loc[i, 'Vegetables[g]'] + count*0.001    
               
#

# Gini Impurity Index for a given hyperrectangle
def calculate_impurity(points):
    return 1 - sum([(np.count_nonzero(points == i)/len(points))**2 for i in np.unique(points)])

#
    
# Split a hyperrectangle dataframe into two based on the best calculated split
def split_dataset(df, split):
    df1 = df.loc[df[split[0]] <= split[1]]
    df2 = df.loc[df[split[0]] > split[1]]
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    return df1, df2

#

# Calculate the best column & value with which to split a dataset
def calculate_best_split(df, var_list, output_var):
    current_impurity = calculate_impurity(df[output_var])
    best_impurity_change = 0
    for i in var_list:
        column = df[i]
        for j in range(len(column) - 1):
            split_threshold = 0.5*(column[j] + column[j + 1])
            new_rect_1 = df.loc[df[i] > split_threshold][output_var]
            new_rect_2 = df.loc[df[i] <= split_threshold][output_var]
            p1 = len(new_rect_1)/(len(new_rect_1) + len(new_rect_2))
            p2 = len(new_rect_2)/(len(new_rect_1) + len(new_rect_2))
            impurity_change = current_impurity - p1*calculate_impurity(new_rect_1) - p2*calculate_impurity(new_rect_2)
            
            if impurity_change > best_impurity_change:
                split = [i, split_threshold]
                best_impurity_change = impurity_change

                
    return split

#

# Makes a prediction on a point using the tree
# Point must be like a dictionary
def tree_prediction(root, df, point, var_list, output_var):
    current_node, current_df = root, df       
    while True:
        if current_node.left == None and current_node.right == None:
            break
        df1, df2 = split_dataset(current_df, current_node.split)
        if point[current_node.split[0]] <= current_node.split[1] and current_node.left != None:
            current_node = current_node.left
            current_df = df1
        elif point[current_node.split[0]] > current_node.split[1] and current_node.right != None:
            current_node = current_node.right 
            current_df = df2
          
    current_df = list(current_df[output_var])
    

    return max(set(current_df), key=current_df.count) 


#

# Test the accuracy of the tree using the test dataframe
def test_accuracy(root, df_train, var_list, output_var, df_test):
    count = 0
    for i in range(len(df_test)):
        point = df_test.iloc[i]
        prediction = tree_prediction(root, df_train, point, var_list, output_var)
        ground_truth = point[output_var]
        if prediction == ground_truth:
            count += 1           
            
    return count/len(df_test)
                
#

# Returns the number of leaf nodes in a tree. Leaf count has to be a list
# Because lists, unlike integers, are mutable, so the fn can modify them        
def count_leaves(node):
    leaf_count = [0]
    def count_leaves(node): 
        if node.left != None and node.right != None:
            if node.left.left == None and node.left.right == None:
                leaf_count[0] += 1
            else:
                count_leaves(node.left)            
            if node.right.left == None and node.right.right == None:
                leaf_count[0] += 1 
            else:
                count_leaves(node.right)
                
    count_leaves(node)
                
    return leaf_count[0]

#                
           
# Remove branches from the tree if the new tree is more accurate on the test set
def prune(node, df_train, var_list, output_var, df_test, root_node, alpha):
    if node.left != None and node.right != None:
        inaccuracy_before = 1 - test_accuracy(root_node, df_train, var_list, output_var, df_test)
        leaf_count_before = count_leaves(root_node)
        
        temp_left, temp_right = node.left, node.right
        node.left, node.right = None, None
        
        inaccuracy_after = 1 - test_accuracy(root_node, df_train, var_list, output_var, df_test)
        leaf_count_after = count_leaves(root_node)
        
        penalty_before = inaccuracy_before + alpha*leaf_count_before
        penalty_after = inaccuracy_after + alpha*leaf_count_after
        if penalty_after > penalty_before:
            node.left, node.right = temp_left, temp_right
            prune(node.left, df_train, var_list, output_var, df_test, root_node, alpha) 
            prune(node.right, df_train, var_list, output_var, df_test, root_node, alpha) 
            
            
# Functions for the Forest
#____________________________
 
# Take the mode of the predictions of the individual trees in the forest           
def forest_prediction(forest, point, var_list, output_var):
    predictions = []
    for i in forest:
        predictions.append(tree_prediction(i[0], i[1], point, var_list, output_var))
     
    return max(set(predictions), key=predictions.count)  

#
# Test the accuracy of the tree using the test dataframe
def forest_accuracy(forest, df_test, var_list, output_var):
    count = 0
    for i in range(len(df_test)):
        point = df_test.iloc[i]
        prediction = forest_prediction(forest, point, var_list, output_var)
        ground_truth = point[output_var]
        if prediction == ground_truth:
            count += 1           
            
    return count/len(df_test)
