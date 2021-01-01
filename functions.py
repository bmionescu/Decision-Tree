
# Functions for the Decision tree

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#____________________________

# Gini Impurity Index for a given hyperrectangle
def calculate_impurity(points):
    return 1 - sum([(np.count_nonzero(points == i)/len(points))**2 for i in np.unique(points)])

# Calculate the best way to split a hyperrectangle
def calculate_best_split(df, var_list, output_var):
    current_impurity = calculate_impurity(df[output_var])
    best_impurity_change = 0
    for i in var_list:
        column = df[i]
        for j in range(len(column)-1):
            split_threshold = 0.5*(column[j] + column[j + 1])
            new_rect_1 = [df[output_var][i] for i in range(len(column)) if column[i] > split_threshold]
            new_rect_2 = [df[output_var][i] for i in range(len(column)) if column[i] <= split_threshold]
            p1 = len(new_rect_1)/(len(new_rect_1) + len(new_rect_2))
            p2 = len(new_rect_2)/(len(new_rect_1) + len(new_rect_2))
            impurity_change = current_impurity - p1*calculate_impurity(new_rect_1) - p2*calculate_impurity(new_rect_2)
            
            if impurity_change > best_impurity_change:
                split = [i, split_threshold]
                best_impurity_change = impurity_change
                
    return split

# Split a hyperrectangle dataframe into two based on the best calculated split
def split_dataset(df, split, var_list, output_var):
    new_rect_1, new_rect_2 = [], []
    columns = var_list.copy()
    columns.append(output_var)
    for i in range(len(df)):
        data_point = [df[j][i] for j in columns]
        if df[split[0]][i] <= split[1]:
            new_rect_1.append(data_point)
        else:
            new_rect_2.append(data_point)
                  
    df1 = pd.DataFrame(new_rect_1, columns = columns)
    df2 = pd.DataFrame(new_rect_2, columns = columns)
    
    return df1, df2

# Makes a prediction on a point using the tree
# Point must be like a dictionary
def check_point(root, df, point, var_list, output_var):
    current_node, current_df = root, df       
    while True:
        if current_node.left == None and current_node.right == None:
            break
        df1, df2 = split_dataset(current_df, current_node.split, var_list, output_var)
        if point[current_node.split[0]] <= current_node.split[1] and current_node.left != None:
            current_node = current_node.left
            current_df = df1
        elif point[current_node.split[0]] > current_node.split[1] and current_node.right != None:
            current_node = current_node.right 
            current_df = df2
#        else:
#            print(current_node.split, point)
            
    current_df = list(current_df[output_var])
    
    return max(set(current_df), key=current_df.count)  

# Test the accuracy of the tree using the test dataframe
def test_accuracy(root, df_train, var_list, output_var, df_test):
    count = 0
    for i in range(len(df_test)):
        point = df_test.iloc[i]
        prediction = check_point(root, df_train, point, var_list, output_var)
        ground_truth = point[output_var]
        if prediction == ground_truth:
            count += 1           
            
    return count/len(df_test)
                
#

# Returns the number of leaf nodes in a tree. Leaf count has to be a list
# Because of some Python mechanics           
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

#______________________________________________________
            
# Recursively plot dividing lines on the plot to show the tree's different categories
def plot_dividing_lines(df, node, var_list, output_var):
    if node.left != None and node.right != None:
        df1, df2 = split_dataset(df, node.split, var_list, output_var)
        axis = df[get_axis(node.split[0])] 
        if node.split[0] == 'x1':
            plt.vlines(node.split[1], min(axis), max(axis), linestyles='dotted')
        if node.split[0] == 'x2':
            plt.hlines(node.split[1], min(axis), max(axis), linestyles='dotted') 
        
        plot_dividing_lines(df1, node.left, var_list, output_var)
        plot_dividing_lines(df2, node.right, var_list, output_var)    

            
# Plot the data
def plot(df, root_node, var_list, output_var):
    plt.figure(figsize = (8, 6)) 
    for i in range(len(df)):
        if df[output_var][i] == 1:
            mark = 'r+'
        if df[output_var][i] == 2:
            mark = 'b+'        
        if df[output_var][i] == 3:
            mark = 'g+'        
        if df[output_var][i] == 4:
            mark = 'm+'
        
        plt.plot([df[var_list[0]][i]], [df[var_list[1]][i]], mark)
    
    plot_dividing_lines(df, root_node, var_list, output_var)
    a, b, c, d = min(list(df['x1'])), max(list(df['x1'])), min(list(df['x2'])), max(list(df['x2']))
    plt.axis([a - 0.5, b + 0.5, c - 4, d + 4])
    plt.show()


#
def get_axis(string):
    if string == 'x1':
        return 'x2'
    else:
        return 'x1'
    