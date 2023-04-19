from copy import deepcopy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
#import decision_tree
from decision_tree import decision_tree
import misc



class random_forest:
    def __init__(self, data: list, num_trees: int, attr_type: list, attr_labels: list, stopping_criteria = "minimal_gain_criterion"):
        # list containing all trees that are members of the forest
        self.trees = []
        # helper dictionary to translate between category labels and (column) indices
        self.cat_to_attr_index = {}
        for index in range(len(data[0])):
            self.cat_to_attr_index[data[0][index]] = index

        # helper 2-D list to hold all possible values for given categorical attributes
        self.attr_vals = {}
        for index in range(len(data[0]) - 1): # -1 to avoid copying the "target" attribute
            self.attr_vals[data[0][index]] = list()
            self.attr_vals[index] = list()

        for index in range(len(data[0]) - 1): # -1 to avoid copying the "target" attribute
            if attr_type[index] == False: # only categorical attributes need their values stored
                for entry in range(1, len(data)): # start at 1 to avoid the labels in the first row
                    if data[entry][index] not in self.attr_vals[index]:
                        self.attr_vals[index].append(data[entry][index])
                        self.attr_vals[data[0][index]].append(data[entry][index])

        # Now that we're done creating those helper data structures, let's remove the labels 
        # from the first row of the dataset
        data.pop(0)

        for _ in range(num_trees):
            tree_data = misc.bootstrap(data)
            self.trees.append(decision_tree.decision_tree(deepcopy(tree_data), None,\
               stopping_criteria, attr_type, attr_labels, deepcopy(self.attr_vals), '', is_root=True, split_metric="Info_Gain"))

    def classify_instance(self, instance: list, attr_type):
        votes = {}
        # collect the votes from all of the member trees
        for worker in self.trees: 
            vote = worker.classify_instance(instance, self.cat_to_attr_index, attr_type)
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1
        # and return the classification with the most votes
        #print(f"votes: {votes}")
        return max(votes, key = votes.get)

    def recur_print(self, tree_index=-1):
        if tree_index == -1:
            for i in range(len(self.trees)):
                print(f"Tree {i}:")
                self.trees[i].recursive_print()
        else:
            self.trees[tree_index].recursive_print()

#def main():
#    print("Hey")

#if __name__ == "__main__":
#    main()
