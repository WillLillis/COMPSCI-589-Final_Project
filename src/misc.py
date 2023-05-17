# in an effort to keep this project more manageable, I'll reorganize and throw
# some of the utility functions in here...

# assignment asks us to define a boostrap method to pass data to the trees of the random forest
# assignment then says we'll test the forest via kfold cross validation, so I guess we don't have to keep track of 
# our "out of bag" instances???
from copy import deepcopy
import csv
import numpy as np
import random
import sys
import os
from knn import knn
from neural_net import neural_net
from random_forest import random_forest

# returns accuracy, precision, recall, and f1 score
def get_metrics(labels: list, preds: list, num_classes: int):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(preds)):
        conf_matrix[labels[i]][preds[i]] += 1

    #print(f"{conf_matrix=}")
    # https://stackoverflow.com/questions/40729875/calculate-precision-and-recall-in-a-confusion-matrix
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    true_pos = np.diag(conf_matrix)
    false_pos = np.sum(conf_matrix, axis=0) - true_pos
    false_neg = np.sum(conf_matrix, axis=1) - true_pos

    precision = np.nanmean(true_pos / (true_pos + false_pos))
    recall = np.nanmean(true_pos / (true_pos + false_neg))
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


def learning_curve_knn(training_set, test_set, num_neighbors, num_classes):
    num_k = [num for num in range(1, 51) if num % 5 == 0 or num == 1]

    for k in num_k:
        accuracy, _, _, f1_score = knn.knn_test(training_set, test_set, k, num_classes)
        # print(f'{k=}')
        print(f'{accuracy}')
        # print(f'{f1_score}')

def learning_curve_nn(regularization, net_shape, training_set, test_set, num_classes):
    end = 5
    while end <= len(training_set):
        temp_train_set = deepcopy(training_set)[:end]
        end += 5
        accuracy, _, _, f1_score = neural_net.main(regularization, net_shape, np.array(temp_train_set), np.array(test_set), num_classes)
        print(f'{end=}')
        print(f'{accuracy=}')
        print(f'{f1_score=}')

def learning_curve_rf(training_set, test_set, attr_type, data_labels_num, num_classes):
    num_trees = [num for num in range(1, 51) if num % 5 == 0 or num == 1]

    for tree in num_trees:
        accuracy, _, _, f1_score = random_forest.main(training_set, test_set, tree, attr_type, data_labels_num, num_classes)
        print(f'{tree=}')
        print(f'{accuracy=}')
        print(f'{f1_score=}')

# returns a bootstrap of the data set passed in, with labels still in the first row
def bootstrap(data: list):
    length = len(data)
    strap = list()
    #strap.append(deepcopy(data[0])) # keep the labels up top

    for _ in range(length - 1):
        strap.append(deepcopy(data[random.randrange(1, length - 1)]))
    return strap

# generates the 
def k_folds_gen(k: int, file_name: str, normalize_attrs: bool):
    # find class proportiions in data set
    # make k folds
    # populate each fold according to class proportions (randomly)
    # for each fold i...
        # pass training data in to train random forest
        # evaluate on fold i

    # create numeric/ categorical attribute list on load time

    # True for numeric, False for categorical
    attr_type = list()
    with open(os.path.join(os.path.dirname(__file__), os.pardir, os.path.join('datasets',file_name)), encoding="utf-8") as raw_data_file: 
    #with open(file_name, encoding="utf-8") as raw_data_file:
        # the data files all follow different labeling conventions and/or use different delimiters...
        # could make this more general, but here we'll more or less hardcode in the correct procedure for
        # the cancer, house votes, and wine datasets
        if 'hw3_cancer.csv' in file_name:
            data_reader = csv.reader(raw_data_file, delimiter='\t')
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                for j in range(len(data_set[i])):
                    data_set[i][j] = int(float(data_set[i][j]))
            for _ in range(len(data_set[0]) - 1):
                attr_type.append(True)
        elif 'hw3_house_votes_84.csv' in file_name:
            data_reader = csv.reader(raw_data_file)
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                for j in range(len(data_set[i])):
                    data_set[i][j] = int(data_set[i][j])
            for _ in range(len(data_set[0]) - 1):
                attr_type.append(False)
        elif 'hw3_wine.csv' in file_name:
            data_reader = csv.reader(raw_data_file, delimiter='\t')
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                data_set[i][0] = int(data_set[i][0])
                for j in range(1, len(data_set[i])):
                    data_set[i][j] = float(data_set[i][j])
            # for the sake of simplicity, at this point I'm going to move
            # the wine classes to the last column so it matches the other data sets
            for entry in data_set:
                tmp = entry.pop(0)
                entry.append(tmp)
            for _ in range(len(data_set[0]) - 1):
                attr_type.append(True)
        elif 'loan.csv' in file_name:
            data_reader = csv.reader(raw_data_file)
            data_set = list(data_reader)
            # throw out the loan id attribute
            for i in range(len(data_set)):
                data_set[i].pop(0)
            # cast attribute values to appropriate data types from strings
            for i in range(1, len(data_set)):
                for j in range(len(data_set[0])):
                    if j == 0: # Gender: from 'Male'/'Female' to 0/1
                        data_set[i][j] = 0 if data_set[i][j] == 'Male' else 1
                    elif j == 1: # Married: from 'No'/'Yes' to 0/1
                        data_set[i][j] = 0 if data_set[i][j] == 'No' else 1
                    elif j == 2: # Dependents: from '0','1','2','3+' to 0,1,2,3
                        data_set[i][j] = 3 if data_set[i][j] == '3+' else (int(data_set[i][j]))
                    elif j == 3: # Education: from 'Not Graduate'/'Graduate' to 0/1
                        data_set[i][j] = 0 if data_set[i][j] == 'Not Graduate' else 1
                    elif j == 4: # Self_Employed: from 'No'/'Yes' to 0/1
                        data_set[i][j] = 0 if data_set[i][j] == 'No' else 1
                    elif j == 5 or j == 6 or j == 7 or j == 8 or j == 9: # ApplicantIncome or CoapplicantIncome or Loan_Amount or Loan_Amount_Term or Credit_History
                        data_set[i][j] = int(float(data_set[i][j]))
                    elif j == 10: # Property_Area: from 'Rural'/'Semiurban'/'Urban' to 0/1/2
                        if data_set[i][j] == 'Rural':
                            data_set[i][j] = 0
                        elif data_set[i][j] == 'Semiurban':
                            data_set[i][j] = 1
                        elif data_set[i][j] == 'Urban':
                            data_set[i][j] = 2
                    elif j == 11:
                        data_set[i][j] = 0 if data_set[i][j] == 'N' else 1
            attr_type.append(False) # Gender
            attr_type.append(False) # Married
            attr_type.append(False) # Dependents
            attr_type.append(False) # Education
            attr_type.append(False) # Self_Employed
            attr_type.append(True)  # ApplicantIncome
            attr_type.append(True)  # Coapplicant Income
            attr_type.append(True)  # LoanAmount
            attr_type.append(True)  # Loan_Amount_Term
            attr_type.append(False) # Credit_History
            attr_type.append(False) # Property_Area
            # Normalize features if specified (ASSUMES STRICTLY NON-NEGATIVE VALUES)
            if normalize_attrs:
                for i in range(len(attr_type)):
                    if attr_type[i]: # if it's a numerical attribute
                        tmp_max = 0
                        for j in range(1, len(data_set)): # Find the max value for the given numerical attribute
                            tmp_max = max(tmp_max, data_set[j][i])
                        for j in range(1, len(data_set)): # Scale all the values according to this max value
                            data_set[j][i] /= tmp_max
        elif 'parkinsons.csv' in file_name:
            pass
            # all attributes (except the class label) are floats
            # negative values included, so we need to use the more generalized normalization formula
            data_reader = csv.reader(raw_data_file)
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                data_set[i][-1] = int(data_set[i][-1])
                for j in range(len(data_set[0]) - 1):
                    data_set[i][j] = float(data_set[i][j])
            for i in range(len(data_set[0]) - 1):
                attr_type.append(True)
            if normalize_attrs:
                for i in range(len(attr_type)):
                    tmp_max = float('-inf')
                    tmp_min = float('inf')
                    for j in range(1, len(data_set)): # Find the max value for the given numerical attribute
                        tmp_max = max(tmp_max, data_set[j][i])
                        tmp_min = min(tmp_min, data_set[j][i])
                    for j in range(1, len(data_set)):
                        data_set[j][i] = (data_set[j][i] - tmp_min) / (tmp_max - tmp_min)
        elif 'titanic.csv' in file_name:
            data_reader = csv.reader(raw_data_file)
            data_set = list(data_reader)
            # throw out the passenger name attribute
            for i in range(len(data_set)):
                data_set[i].pop(2)
            # put the class labels in the last column
            for i in range(len(data_set)):
                tmp = data_set[i].pop(0)
                data_set[i].append(tmp)
            # cast attribute values to appropriate data types from strings
            for i in range(1, len(data_set)):
                for j in range(len(data_set[0])):
                    if j == 0: 
                        data_set[i][j] = int(data_set[i][j]) - 1
                    elif j == 1: # translate 'male'/'female' to '0'/'1'
                        data_set[i][j] = 0 if data_set[i][j] == 'male' else 1
                    elif j == 2 or j == 5: # age and fare both floats
                        data_set[i][j] = float(data_set[i][j])
                    else: # rest ints
                        data_set[i][j] = int(data_set[i][j])
            attr_type.append(False) # Pclass
            attr_type.append(False) # Sex
            attr_type.append(True)  # Age
            attr_type.append(False) # Siblings/Spouses Aboard
            attr_type.append(False) # Parents/Children Aboard
            attr_type.append(True)  # Fare
            if normalize_attrs:
                for i in range(len(attr_type)):
                    if attr_type[i]: # if it's a numerical attribute
                        tmp_max = 0
                        for j in range(1, len(data_set)): # Find the max value for the given numerical attribute
                            tmp_max = max(tmp_max, data_set[j][i])
                        for j in range(1, len(data_set)): # Scale all the values according to this max value
                            data_set[j][i] /= tmp_max
        elif 'optdigits' in file_name:
            data_reader = csv.reader(raw_data_file)
            data_set = list(data_reader)
            # cast attribute values to appropriate data types from strings
            for i in range(len(data_set)):
                for j in range(len(data_set[0])):
                    data_set[i][j] = int(data_set[i][j])
            # TODO: do this for 64 attributes lol - check optdigits.names
            for i in range(64):
                attr_type.append(True)
            if normalize_attrs:
                for i in range(len(attr_type)):
                    if attr_type[i]: # if it's a numerical attribute
                        for j in range(len(data_set)):
                            data_set[j][i] /= 16
                        #tmp_max = float('-inf')
                        #tmp_min = float('inf')
                        #for j in range(len(data_set)): # Find the max value for the given numerical attribute
                        #    tmp_max = max(tmp_max, data_set[j][i])
                        #    tmp_min = min(tmp_min, data_set[j][i])
                        #for j in range(len(data_set)): # Scale all the values according to this max value
                        #    data_set[j][i] = (data_set[j][i] - tmp_min) / (tmp_max - tmp_min)
            #print(f"Dataset:\n\n\n{data_set}")
        else:
            print(f"Bad file name passed as parameter! ({file_name})")
            return None

    class_partitioned = {}
    for i in range(1, len(data_set)):
        if data_set[i][-1] in class_partitioned:
            class_partitioned[data_set[i][-1]].append(data_set[i])
        else:
            class_partitioned[data_set[i][-1]] = list()
            class_partitioned[data_set[i][-1]].append(data_set[i])
    
    class_proportions = {}
    for item in class_partitioned:
        class_proportions[item] = len(class_partitioned[item]) / (len(data_set) - 1)


    for _, cls in class_partitioned.items():
        arr = np.array(cls)
        np.random.shuffle(arr)
        cls = list(arr)
    fold_size = int((len(data_set)-1) / k)
    folds = []
    for i in range(k):
        fold = []
        for idx, cls in class_partitioned.items():
            ratio = len(class_partitioned[idx])/(len(data_set)-1)
            fold_range = int(fold_size * ratio)+1
            indices = cls[int(i*fold_range):int((i+1)*fold_range)]
            for row in indices:
                fold.append(row)
        folds.append(fold)

    return folds, attr_type, deepcopy(data_set[0])

    # populate the folds according to the original data set's class proportions
    # ok to do this in a randomized fashion?

    
    