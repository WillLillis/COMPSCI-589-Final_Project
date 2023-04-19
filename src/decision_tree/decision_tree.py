from copy import deepcopy
import csv
from random import randrange
from math import log2
import random
from statistics import stdev
#import matplotlib.pyplot as plt
from math import sqrt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

class decision_tree:
    # data - The dataset passed in to create further nodes with
    # depth - recursive depth in terms of '\t' characters, used for debugging purposes
    # is_root - indicates whether a node is the root node, must be passed in as true for 
    #           the initial call, false for the other calls
    # classification - has argument passed in only when the caller is passing it an 
    #                  empty data set, indicates what classification (0 or 1 in the case) to make the resulting leaf node
    def __init__(self, data: list, attr_val, stopping_criteria: str, attr_type: list, attr_labels: list, \
        attr_vals: dict, depth = '', is_root = False, classification = None, split_metric="Info_Gain"):
        self.children = []
        self.threshold = None # holds threshold if we end up splitting based off of a numerical attribute
        if is_root == True:
            self.attr_val = None 
        else:
            self.attr_val = attr_val
        # first check if an empty dataset was passed in 
        if classification != None: # determined by caller, != None only when an empty dataset is passed in
            self.is_leaf = True
            self.node_attr = None
            self.classification = classification 
            return

        self.is_leaf = False 
        self.classification = None # node isn't a leaf, so it doesn't decide what class a given instance belongs to 
        split_attr = None

        # evaluate stopping criteria
        if stopping_criteria == "minimal_size_for_split_criterion":
            thresh = 15 # empirical threshold
            if (len(data)) <= thresh:
                self.is_leaf = True
                self.node_attr = None
                self.classification = get_majority_class(data)
                return
            else:
                self.is_leaf = False
        elif stopping_criteria == "minimal_gain_criterion":
             thresh = 0.1 # arbitary threshold
             info_gains = {} # information gain for each attribute
             data_set_only_labels = [] # strip off just the class labels (0's and 1's) to calculate entropy/ info gain
             for index in range(len(data)):
                 data_set_only_labels.append(deepcopy(data[index][-1]))

             # get random subset of dataset's attributes
             attributes = get_rand_cats(deepcopy(attr_labels))
             for i in range(len(attributes)):
                 if attr_type[i] == True: # if it's a numerical attribute...
                     partitions, _ = partition_data_numerical(data, attributes[i], attr_labels) # paritition 'data' according to the current attribute 'attr'
                 else: # otherwise it's categorical
                     partitions = partition_data_categorical(data, attributes[i], attr_vals, attr_labels) # paritition 'data' according to the current attribute 'attr'
                 info_gains[attributes[i]] = info_gain(data_set_only_labels, partitions) 
             split_attr = max(info_gains, key = info_gains.get) # get the attribute of maximal gain

             if info_gains[split_attr] < thresh:
                self.is_leaf = True
                self.node_attr = None
                self.classification = get_majority_class(data)
                return
             else:
                self.is_leaf = False
                self.node_attr = split_attr
        elif stopping_criteria == "maximal_depth_stopping_criterion":
            thresh = 10
            if len(depth) >= thresh:
                self.is_leaf = True
                self.node_attr = None
                self.classification = get_majority_class(data)
                return
            else:
                self.is_leaf = False
        else:
            print(f"Error! Invalid stopping criteria argument provided! ({stopping_criteria})")
            return 

        # find best attribute to split off of based off of Information Gain/ Gini Metric
        if split_metric == "Info_Gain":
            # if we're using minimal gain as our stopping criteria, everything should already be calculated!
            if stopping_criteria == "minimal_gain_criterion": 
                pass
            # otherwise calculate info gain as normal
            else:
                thresh = 0.1 # arbitary threshold
                info_gains = {} # information gain for each attribute
                data_set_only_labels = [] # strip off just the class labels (0's and 1's) to calculate entropy/ info gain
                for index in range(len(data)):
                    data_set_only_labels.append(deepcopy(data[index][-1]))

                # get random subset of dataset's attributes
                attributes = get_rand_cats(deepcopy(attr_labels))
                for i in range(len(attributes)):
                    if attr_type[i] == True: # if it's a numerical attribute...
                        partitions, _ = partition_data_numerical(data, attributes[i], attr_labels) # paritition 'data' according to the current attribute 'attr'
                    else: # otherwise it's categorical
                        partitions = partition_data_categorical(data, attributes[i], attr_vals, attr_labels) # paritition 'data' according to the current attribute 'attr'
                    info_gains[attributes[i]] = info_gain(data_set_only_labels, partitions) 
                split_attr = max(info_gains, key = info_gains.get) # get the attribute of maximal gain

                if info_gains[split_attr] < thresh:
                    self.is_leaf = True
                    self.node_attr = None
                    self.classification = get_majority_class(data)
                    return
                else:
                    self.is_leaf = False
                    self.node_attr = split_attr                
        elif split_metric == "Gini":
            ginis = {}
            data_set_only_labels = []
            for index in range(len(data)):
                data_set_only_labels.append(deepcopy(data[index][-1]))

            # get random subset of dataset's attributes
            attributes = get_rand_cats(deepcopy(attr_labels))
            for attr in attributes:
                ginis[attr] = 0
                if attr_type[attr] == True:
                    partitions, _ = partition_data_numerical(data, attr, attr_labels) # paritition 'data' according to the current attribute 'attr'
                else:
                    partitions = partition_data_categorical(data, attr, attr_vals, attr_labels) # paritition 'data' according to the current attribute 'attr'
                for partition in partitions:
                    ginis[attr] += (len(partition)/ len(data)) * gini_criterion(partition)
            split_attr = min(ginis, key = ginis.get)
        else:
            print("ERROR: Invalid split metric supplied!")
            return
        
        self.node_attr = split_attr
        # partition data based off of split_attr
        if attr_type[i] == True: # if it's a numerical attribute...
            child_data, self.threshold = partition_data_numerical(data, split_attr, attr_labels, labels_only=False) # paritition 'data' according to the current attribute 'attr'
        else: # otherwise it's categorical
            child_data = partition_data_categorical(data, split_attr, attr_vals, attr_labels, labels_only=False) # paritition 'data' according to the current attribute 'attr'
        
        for i in range(len(attr_type)):
            if attr_labels[i] == split_attr:
                split_attr_index = i
                break

        if attr_type[split_attr_index] == True: # if it's numerical...
            for i in range(2): # with numerical attributes there's always 2 partitions
                if len(child_data[i]) <= 1:
                    num_zero = 0
                    num_one = 0
                    for instance in data:
                        if instance[-1] == 0:
                            num_zero +=  1
                        else:
                            num_one += 1
                    majority = 0 if num_zero >= num_one else 1
                    break
        else: # otherwise it's categorical
            for i in range(len(child_data)):
                if len(child_data[i]) <= 0:
                    num_zero = 0
                    num_one = 0
                    for instance in data:
                        if instance[-1] == 0:
                            num_zero +=  1
                        else:
                            num_one += 1
                    majority = 0 if num_zero >= num_one else 1
                    break
        if len(depth) > 10:
            print(f"{depth}{child_data=}")

        if attr_type[split_attr_index] == True: # if it's numerical...
            # could be smarter about this, but I'm just going to hard code it...
            # dictionary doesn't get used at all, but feels better doing this than passing 'None'
            tmp_attr_val = {}
            tmp_attr_val["type"] = "numerical"
            tmp_attr_val["value"] = "leq" # less than or equal to partition
            if len(child_data[0]) > 1:
                self.children.append(decision_tree(child_data[0], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', split_metric=split_metric))
            else:
                self.children.append(decision_tree(child_data[0], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', classification=majority, split_metric=split_metric))
            tmp_attr_val["value"] = "g" # greater than partition
            if len(child_data[1]) > 1:
                self.children.append(decision_tree(child_data[1], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', split_metric=split_metric))
            else:
                self.children.append(decision_tree(child_data[1], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', classification=majority, split_metric=split_metric))
        else: # otherwise it's categorical
            #tmp_attr_val = {}
            #tmp_attr_val["type"] = "categorical"
            for i in range(len(child_data)):
                #tmp_attr_val["value"] = i
                #if len(child_data[i]) > 1:
                if len(child_data[i]) > 0:
                    self.children.append(decision_tree(child_data[i], i, stopping_criteria, attr_type, attr_labels,\
                        attr_vals, depth=depth + '\t', split_metric=split_metric))
                else:
                    self.children.append(decision_tree(child_data[i], i, stopping_criteria, attr_type, attr_labels,\
                        attr_vals, depth=depth + '\t', classification=majority, split_metric=split_metric))
    
    # for debugging, conducts a DFS of the tree, printing out its attributes
    def recursive_print(self, depth=''):
        print(f'{depth}self.attr_val: {self.attr_val}')
        print(f'{depth}self.is_leaf: {self.is_leaf}')
        print(f'{depth}self.node_attr: {self.node_attr}')
        print(f'{depth}self.classification: {self.classification}')
        for child in self.children:
            child.recursive_print(depth + '\t')
    
    # classifies data 'instance' using the current tree
    def classify_instance(self, instance, attr_to_index, attr_type):
        if self.is_leaf == True: # base case
            return self.classification
        if attr_type[self.node_attr] == True: # if it's numerical...
            if instance[self.node_attr] <= self.threshold:
                return self.children[0].classify_instance(instance, attr_to_index, attr_type)
            else:
                return self.children[1].classify_instance(instance, attr_to_index, attr_type)
        else: # otherwise it's categorical
            for child in self.children:
                if child.attr_val == instance[self.node_attr]: # get instance's value for the current node's 'self.node_attr'-> has to match in value with one of children
                    return child.classify_instance(instance, attr_to_index, attr_type)
            
            print(f'BAD ATTRIBUTE LABEL! ({self.node_attr})')
            return None

# partition dataset 'data' based off of attribute 'attr'
# if labels_only=True-> returns partitions ONLY WITH CLASS LABELS (0's and 1's, that's it)
# if labels_only=False-> returns the partitions with entire rows from data set copied in
    # in this case, each partition gets a row of attribute labels at the top
# pass in attr to index dict?
def partition_data_categorical(data, attr, attr_vals: dict, attr_labels: list, labels_only=True)->list: 
    #print("PARTITION!!!!")
    #print(f"attr: {attr}")
    #print(f"attr_vals: {attr_vals}")
    #print(f"attr_labels: {attr_labels}")

    partitions = [] # creating multi-dimensional arrays in python is weird...
    for _ in range(len(attr_vals[attr])):
        partitions.append([])

    for i in range(len(attr_labels) - 1):
        if attr_labels[i] == attr:
            attr_index = i
            break
    # going to abuse the fact that the categorical attribute values are 0,1,2,... and use them as indices in the partitions list
        # using value of attribute for index into partition list (array?)
    # if they weren't I could just use a dict to map the value to an index, but that looks messy....
    if labels_only == True:
         for i in range(len(data)):
            partitions[data[i][attr_index]].append(deepcopy(data[i][-1]))
    else:
        for i in range(len(data)):
            partitions[data[i][attr_index]].append(deepcopy(data[i]))

    return partitions

# BUGBUG have this return the average value used for the split????
def partition_data_numerical(data, attr, attr_labels: list, labels_only=True)->list:
    # always split to two classes based on threshold with numerical attributes...
    partitions = []
    partitions.append([]) # <= partition
    partitions.append([]) # > partition

    # for now we'll just use the "average" approach, will go back and try out the in between approach later
    for i in range(len(attr_labels) - 1):
        if attr_labels[i] == attr:
            attr_index = i
            break
    # grab the average value....
    avg = 0
    for i in range(len(data)):
        avg += data[i][attr_index]
    avg /= (len(data) - 1) # could check before we potentially divide by 0....

    if labels_only == True:
        for i in range(len(data)):
            if data[i][attr_index] <= avg:
                partitions[0].append(deepcopy(data[i][-1]))
            else:
                partitions[1].append(deepcopy(data[i][-1]))
    else:
        for i in range(len(data)):
            if data[i][attr_index] <= avg:
                partitions[0].append(deepcopy(data[i]))
            else:
                partitions[1].append(deepcopy(data[i]))

    return partitions, avg


# only pass in class labels
def entropy(data_set):
    counts = {}
    entropy = 0
    set_size = len(data_set)
    if set_size == 0: # special case, the empty set has an entropy of 0 by definition
        return 0
    
    # count how many of each value we have in 'data_set'
    for entry in data_set:
        if entry in counts: # if it's already been added to the dict, increment the count
            counts[entry] += 1
        else: # otherwise add it to the dict with an initial value of 1
            counts[entry] = 1

    for label in counts:
        entropy += (-counts[label] / set_size) * log2(counts[label] / set_size) # entropy formula...
    return entropy

# orig_data is the original data set, data_split is a tuple of that set partitioned according to some attribute
# assuming we're just using average entropy here...
def info_gain(orig_data, data_split):
    entropies = []
    for split in data_split:
        entropies.append((entropy(split), len(split)))
    
    avg_entropy = 0
    for entry in entropies:
        if len(orig_data) > 0: # avoid dividing by 0, and empty set should have an entropy of 0 so skipping the addition is fine (not adding anything is the same as adding 0)
            avg_entropy += (entry[1] / len(orig_data)) * entry[0]  # weighted average-> (size of partition/ size of set) * entropy of partition

    return entropy(orig_data) - avg_entropy 

# only pass in class labels
# want minimum gini criterion      
def gini_criterion(labels):
    counts = {}
    for x in labels:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1

    accum = 0
    for x in counts:
        accum += (x/len(labels))**2

    return 1 - accum

# just needs to return list of attribute labels
# could also return indices?
# just pass in the first row of the dataset with the labels AS A DEEPCOPY
    # this function will destructively modify its `cats` parameter
# assuming the classification label is included here just for simplicity
# if num_cats_req is specified, we'll use that number
# otherwise we'll use the sqrt heuristic
def get_rand_cats(cats: list, num_cats_req=0):
    ret_cats = list()
    num_cats = 0
    if num_cats_req <= 0:
        num_cats = max(int(sqrt(len(cats) - 1)), 1)
    else:
        num_cats = num_cats_req
        while num_cats > (len(cats) - 1):
            num_cats -= 1
    for _ in range(num_cats):
        ret_cats.append(cats.pop(random.randrange(len(cats) - 1)))
    return ret_cats

def get_majority_class(data: list):
    if len(data) < 1:
        #print("ERROR: Bad dataset!")
        return None
    counts = {}
    #for i in range(1, len(data)):
    for i in range(len(data)):
        if data[i][-1] in counts:
            counts[data[i][-1]] += 1
        else:
            counts[data[i][-1]] = 1
    return max(counts, key=counts.get)

def main():
    pass

if __name__ == '__main__':
    main()