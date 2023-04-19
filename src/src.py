# TODO:
    # Further cleanup of existing code
    # Start testing random forest on new data sets
    # have a look at knn
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
def test_loans(num_folds: int)-> None:
    k_folds, attr_type, attr_labels = misc.k_folds_gen(num_folds, os.path.join("The_Loan_Eligibility_Prediction_Dataset", "loan.csv"), True)

    print(f"attr_type: {attr_type}")
    print(f"attr_labels: {attr_labels}")
    for k in range(len(k_folds)):
        print(f"Fold {k}:")
        print(k_folds[k])
def test_parkinsons(num_folds: int)-> None:
    k_folds, attr_type, attr_labels = misc.k_folds_gen(num_folds, os.path.join("The_Oxford_Parkinson's_Disease_Detection_Dataset", "parkinsons.csv"), True)

    print(f"attr_type: {attr_type}")
    print(f"attr_labels: {attr_labels}")
    for k in range(len(k_folds)):
        print(f"Fold {k}:")
        print(k_folds[k])
def test_titanic(num_folds: int)-> None:
    k_folds, attr_type, attr_labels = misc.k_folds_gen(num_folds, os.path.join("The_Titanic_Dataset", "titanic.csv"), True)

    print(f"attr_type: {attr_type}")
    print(f"attr_labels: {attr_labels}")
    for k in range(len(k_folds)):
        print(f"Fold {k}:")
        print(k_folds[k])
def test_wine(num_trees: int, num_folds: int) -> None:
    k_folds, attr_type, data_labels = misc.k_folds_gen(num_folds, 'hw3_wine.csv', False)
    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        data = []
        test_fold = k_folds[k]
        for index in range(num_folds):
            if index != k:
                data += k_folds[index]
        # slap the labels back onto the top of the k_folds list of lists
        data.insert(0, data_labels)
        data_labels_num = deepcopy(data_labels)
        for i in range(len(data_labels_num) - 1):
            data_labels_num[i] = i

        forest = random_forest.random_forest(data, num_trees, attr_type, data_labels_num)

        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0
        precision1 = 0
        precision2 = 0
        precision3 = 0
        recall1 = 0
        recall2 = 0
        recall3 = 0
        F1_1 = 0
        F1_2 = 0
        F1_3 = 0

        TP_1 = 0
        TP_2 = 0
        TP_3 = 0
        TN_1 = 0
        TN_2 = 0
        TN_3 = 0
        FP_1 = 0
        FP_2 = 0
        FP_3 = 0
        FN_1 = 0
        FN_2 = 0
        FN_3 = 0
        for entry in test_fold:
            output = forest.classify_instance(entry, attr_type)
            if entry[-1] == 1: # first class
                if output == 1:
                    TP_1 += 1
                    TN_2 += 1
                    TN_3 += 1
                elif output == 2:
                    FN_1 += 1
                    FP_2 += 1
                    TN_3 += 1
                elif output == 3:
                    FN_1 += 1
                    TN_2 += 1
                    FP_3 += 1
                else:
                    print("wtf")
            elif entry[-1] == 2: # second class
                if output == 1:
                    FP_1 += 1
                    FN_2 += 1
                    TN_3 += 1
                elif output == 2:
                    TN_1 += 1
                    TP_2 += 1
                    TN_3 += 1
                elif output == 3:
                    TN_1 += 1
                    FN_2 += 1
                    FP_3 += 1
                else:
                    print("wtf")
            elif entry[-1] == 3: # third class
                if output == 1:
                    FP_1 += 1
                    TN_2 += 1
                    FN_3 += 1
                elif output == 2:
                    TN_1 += 1
                    FP_2 += 1
                    FN_3 += 1
                elif output == 3:
                    TN_1 += 1
                    TN_2 += 1
                    TP_3 += 1
                else:
                    print("wtf")
        if TP_1 == 0 and TN_1 == 0 and FP_1 == 0 and FN_1 == 0:
            print(f"ERROR: {TP_1=} and {TN_1=} and {FP_1=} and {FN_1=}")
            return
        accuracy1 = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1)
        if TP_2 == 0 and TN_2 == 0 and FP_2 == 0 and FN_2 == 0:
            print(f"ERROR: {TP_2=} and {TN_2=} and {FP_2=} and {FN_2=}")
            return
        accuracy2 = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2)
        if TP_3 == 0 and TN_3 == 0 and FP_3 == 0 and FN_3 == 0:
            print(f"ERROR: {TP_3=} and {TN_3=} and {FP_3=} and {FN_3=}")
            return
        accuracy3 = (TP_3 + TN_3) / (TP_3 + TN_3 + FP_3 + FN_3)
        if TP_1 == 0 and FP_1 == 0:
            print(f"ERROR: {TP_1=} and {FP_1=}")
            return
        precision1 = (TP_1) / (TP_1 + FP_1)
        if TP_2 == 0 and FP_2 == 0:
            print(f"ERROR: {TP_2=} and {FP_2=}")
            return
        precision2 = (TP_2) / (TP_2 + FP_2)
        if TP_3 == 0 and FP_3 == 0:
            print(f"ERROR: {TP_3=} and {FP_3=}")
            return
        precision3 = (TP_3) / (TP_3 + FP_3)
        if TP_1 == 0 and FN_1 == 0:
            print(f"ERROR: {TP_1=} and {FN_1=}")
            return
        recall1 = (TP_1) / (TP_1 + FN_1)
        if TP_2 == 0 and FN_2 == 0:
            print(f"ERROR: {TP_2=} and {FN_2=}")
            return
        recall2 = (TP_2) / (TP_2 + FN_2)
        if TP_3 == 0 and FN_3 == 0:
            print(f"ERROR: {TP_3=} and {FN_3=}")
            return
        recall3 = (TP_3) / (TP_3 + FN_3)
        if precision1 == 0 and recall1 == 0:
            print(f"ERROR: {precision1=} and {recall1=}")
            return
        F1_1 = (2.0 * precision1 * recall1) / (precision1 + recall1)
        if precision2 == 0 and recall2 == 0:
            print(f"ERROR: {precision2=} and {recall2=}")
            return
        F1_2 = (2.0 * precision2 * recall2) / (precision2 + recall2)
        if precision3 == 0 and recall3 == 0:
            print(f"ERROR: {precision3=} and {recall3=}")
            return
        F1_3 = (2.0 * precision3 * recall3) / (precision3 + recall3)
    accuracies.append((accuracy1 + accuracy2 + accuracy3) / 3.0)
    precisions.append((precision1 + precision2 + precision3) / 3.0)
    recalls.append((recall1 + recall2 + recall3) / 3.0)
    F1s.append((F1_1 + F1_2 + F1_3) / 3.0)
    print(f"Wine Results ({num_trees} trees, {num_folds} folds):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

def test_congress(num_trees: int, num_folds: int)-> None:
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
        if TP == 0 and TN == 0 and FP == 0 and FN == 0:
            print(f"ERROR! {TP=} and {TN=} and {FP=} and {FN=}")
            return
        accuracies.append((TP + TN) / (TP + TN + FP + FN))
        if TP == 0 and FP == 0:
            print(f"ERROR! {TP=} and {FP=}")
            return
        precisions.append(TP / (TP + FP))
        if TP == 0 and FN == 0:
            print(f"ERROR! {TP=} and {FN=}")
            return
        recalls.append(TP / (TP + FN))
        F1s.append((2.0 * precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))
    
    print(f"Congressional Results ({num_trees} trees, {num_folds} folds):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

def main():
    #test_loans(10, 10)
    #test_titanic(10, 10)
    #test_parkinsons(10, 10)
    #test_digits(10, 10)
    test_wine(10, 10)
    test_congress(10, 10)

if __name__ == "__main__":
    main()