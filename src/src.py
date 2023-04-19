# TODO:
    # - Clean up existing code
        # - Remove dependence on data labels being in the first row, just consolidate to an argument
    # - Fill in functions below to load respective data sets
    # - Then write testing functions that will train a specified model 
    # on the loaded data and evaluate it using the metrics of interest 
        # - k-fold stratified cross validation with k=10
        # - Accuracy and F1 score
    # Might need to fight with python in adding the parent directory path so you can import things through the folders...
import misc
from copy import deepcopy
import os
from random_forest import random_forest

def test_digits():
    pass
    # digits stored as rows of 64 numbers
    # originally an 8x8 grayscale pixel array, which was flattened
    # I guess values 0->15 indicate white->black for each pixel
    # going to wait to write code to load this until we see how our NN turns out
def test_loans():
    k_folds, attr_type, attr_labels = misc.k_folds_gen(10, os.path.join("The_Loan_Eligibility_Prediction_Dataset", "loan.csv"), True)

    print(f"attr_type: {attr_type}")
    print(f"attr_labels: {attr_labels}")
    for k in range(len(k_folds)):
        print(f"Fold {k}:")
        print(k_folds[k])
def test_parkinsons():
    k_folds, attr_type, attr_labels = misc.k_folds_gen(10, os.path.join("The_Oxford_Parkinson's_Disease_Detection_Dataset", "parkinsons.csv"), True)

    print(f"attr_type: {attr_type}")
    print(f"attr_labels: {attr_labels}")
    for k in range(len(k_folds)):
        print(f"Fold {k}:")
        print(k_folds[k])
def test_titanic():
    k_folds, attr_type, attr_labels = misc.k_folds_gen(10, os.path.join("The_Titanic_Dataset", "titanic.csv"), True)

    print(f"attr_type: {attr_type}")
    print(f"attr_labels: {attr_labels}")
    for k in range(len(k_folds)):
        print(f"Fold {k}:")
        print(k_folds[k])


def main():
    #test_loans()
    #test_titanic()
    #test_parkinsons()
    #test_digits()
    num_folds = 10
    num_trees = 10
    k_folds, attr_type, data_labels = misc.k_folds_gen(num_folds, 'hw3_house_votes_84.csv', False)
    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        data = []
        test_fold = k_folds[k]
        for index in range(num_folds):
            if index != k:
                data += k_folds[index]
        data.insert(0, data_labels)
        data_labels_num = deepcopy(data_labels)
        for i in range(len(data_labels_num) - 1):
            data_labels_num[i] = i

        forest = random_forest.random_forest(data, num_trees, attr_type, data_labels_num)
        
        for entry in test_fold:
            output = forest.classify_instance(entry, attr_type)
            if entry[-1] == 0: # negative class instances
                if output == 0:
                    TN += 1
                elif output == 1:
                    FP += 1
                else:
                    print("wtf")
            elif entry[-1] == 1: # positive class instances
                if output == 0:
                    FN += 1
                elif output == 1:
                    TP += 1
                else:
                    print("wtf")
            else:
                print("wtf")
        accuracies.append((TP + TN) / (TP + TN + FP + FN))
        precisions.append(TP / (TP + FP))
        recalls.append(TP / (TP + FN))
        F1s.append((2.0 * precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))
    
    print(f"Congressional Results ({num_trees} trees, {num_folds} folds):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

if __name__ == "__main__":
    main()