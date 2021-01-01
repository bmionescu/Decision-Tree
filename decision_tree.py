
import pandas as pd

from functions import (
        calculate_impurity,
        calculate_best_split,
        split_dataset,     
        test_accuracy,
        count_leaves,
        prune,
        plot,
)
    
#____________________________
# Data 

df = pd.read_excel('./test5.xlsx') # df for data frame
var_list = ['x1', 'x2'] # Independent variables
output_var = 'y'

df_train = pd.read_excel('./test3.xlsx')
df_test = pd.read_excel('./test4.xlsx')

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
        
#____________________________
        
# Growing the tree

def grow_tree(df, var_list, output_var):        
    root_node = Node(calculate_impurity(df[output_var]))
    current_node = root_node
    current_df = df
      
    stack = []
    
    cond = True
    while cond:  
        best_split = calculate_best_split(current_df, var_list, output_var)   
        df1, df2 = split_dataset(current_df, best_split, var_list, output_var)
        
        left_impurity = calculate_impurity(df1[output_var])
        right_impurity = calculate_impurity(df2[output_var])    
        
        left_node = Node(left_impurity)
        right_node = Node(right_impurity)
        
        current_node.do_split(left_node, right_node, best_split)
        
        if left_impurity == right_impurity == 0:
            if len(stack) == 0:
                cond = False
            else:
                current_node = stack[-1][0]
                current_df = stack[-1][1]
                del stack[-1]
                   
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
    
# Grow a set of trees, prune for various alpha, and finally, pick the best one
trees, accuracies = [], []
alpha = 0 
   
cond = True
while cond:
    root_node = grow_tree(df_train, var_list, output_var)
    prune(root_node, df_train, var_list, output_var, df_test, root_node, alpha)
    acc = test_accuracy(root_node, df_train, var_list, output_var, df_test)
    trees.append(root_node)
    accuracies.append(acc)
    if root_node.left == None and root_node.right == None:
        break
    alpha += 0.02
    #print(alpha, count_leaves(root_node), acc)

final_model = trees[accuracies.index(max(accuracies))]
final_accuracy = max(accuracies)

#____________________________













