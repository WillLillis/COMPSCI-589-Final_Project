# in an effort to keep this project more manageable, I'll reorganize and throw
# some of the utility functions in here...

# assignment asks us to define a boostrap method to pass data to the trees of the random forest
# assignment then says we'll test the forest via kfold cross validation, so I guess we don't have to keep track of 
# our "out of bag" instances???
from copy import deepcopy
import csv
import random
import sys
import os

# returns a bootstrap of the data set passed in, with labels still in the first row
def bootstrap(data: list):
    length = len(data)
    strap = list()
    strap.append(deepcopy(data[0])) # keep the labels up top

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
    with open(os.path.join(os.path.dirname(__file__), os.pardir, os.path.join('data',file_name)), encoding="utf-8") as raw_data_file: 
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
            # throw out the load id attribute
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
                    if j == 1: # translate 'male'/'female' to '0'/'1'
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
        # no idea what to do with the MNIST files, absolute mess
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

    # create list of lists to hold our k folds
    k_folds = []
    for _ in range(k):
        k_folds.append([])

    entries_per_fold = int((len(data_set) - 1) / k)
    while k * entries_per_fold > (len(data_set) - 1):
        entries_per_fold -= 1

    if len(class_proportions) == 2:
        for index in range(k):
            for _ in range(entries_per_fold):
                if random.uniform(0,1) <= class_proportions[0]:
                    if len(class_partitioned[0]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[0]))
                    new_entry = class_partitioned[0].pop(tmp)
                    k_folds[index].append(new_entry)
                else:
                    if len(class_partitioned[1]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[1]))
                    new_entry = class_partitioned[1].pop(tmp)
                    k_folds[index].append(new_entry)
    elif len(class_proportions) == 3:
        for index in range(k):
            for _ in range(entries_per_fold):
                u = random.uniform(0,1)
                if u <= class_proportions[1]:
                    if len(class_partitioned[1]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[1]))
                    new_entry = class_partitioned[1].pop(tmp)
                    k_folds[index].append(new_entry)
                elif (u > class_proportions[1]) and (u <= (class_proportions[1] + class_proportions[2])):
                    if len(class_partitioned[2]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[2]))
                    new_entry = class_partitioned[2].pop(tmp)
                    k_folds[index].append(new_entry)
                else:
                    if len(class_partitioned[3]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[3]))
                    new_entry = class_partitioned[3].pop(tmp)
                    k_folds[index].append(new_entry)
    else:
        print("ERROR!!!!!!!")

    return k_folds, attr_type, deepcopy(data_set[0])

    # populate the folds according to the original data set's class proportions
    # ok to do this in a randomized fashion?

    
    