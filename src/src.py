# TODO:
    # Further cleanup of existing code
    # Start testing random forest on new data sets
    # have a look at knn
import misc
from copy import deepcopy
import os
import numpy as np
from random_forest import random_forest
from neural_net import neural_net
from knn import knn

def test_digits(num_folds: int, num_trees: int) -> None:
    k_folds, attr_type, attr_labels = misc.k_folds_gen(num_folds, os.path.join("The_Hand-Written_Digits_Recognition_Dataset", "optdigits.comb"), True)
    
    # TODO: tune hyper-parameters, add random forest code
    num_neighbors = 5 # arbitrary, need to tune!
    knn_accuracies = []
    knn_precisions = []
    knn_recalls = []
    knn_f1_scores = []

    nn_accuracies = []
    nn_precisions = []
    nn_recalls = []
    nn_f1_scores = []

    rf_accuracies = []
    rf_precisions = []
    rf_recalls = []
    rf_f1_scores = []

    print(f"Beginning test on the MNIST dataset...")
    for k in range(num_folds):
        test_set = k_folds[k]
        training_set = []
        for i in range(1, len(k_folds)):
            if i != k:
                training_set += k_folds[i]

        if k == 0:
            # misc.learning_curve_knn(training_set, test_set, num_neighbors, 10)
            misc.learning_curve_nn(0, [64,30,10,27,10], training_set, test_set, 10)
            training_set.insert(0, attr_labels)
            data_labels_num = deepcopy(attr_labels)
            for i in range(len(data_labels_num) - 1):
                data_labels_num[i] = i
            # misc.learning_curve_rf(training_set, test_set, attr_type, data_labels_num, 10)
        
    #     # print(f"{k=}:")
    #     # print(f"\tTesting KNN:")
    #     accuracy, precision, recall, f1_score = knn.knn_test(training_set, test_set, num_neighbors, 10)
    #     #print(f"KNN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     knn_accuracies.append(accuracy)
    #     knn_precisions.append(precision)
    #     knn_recalls.append(recall)
    #     knn_f1_scores.append(f1_score)

    #     # print(f"\tTesting NN:")
    #     accuracy, precision, recall, f1_score = neural_net.main(0, [64,30,10,27,10], np.array(training_set), np.array(test_set), 10)
    #     #print(f"NN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     nn_accuracies.append(accuracy)
    #     nn_precisions.append(precision)
    #     nn_recalls.append(recall)
    #     nn_f1_scores.append(f1_score)

    #     # print(f"\tTesting RF:")
    #     # slap the labels back onto the top of the k_folds list of lists
    #     # TODO: ideally get rid of this
    #     training_set.insert(0, attr_labels)
    #     data_labels_num = deepcopy(attr_labels)
    #     for i in range(len(data_labels_num) - 1):
    #         data_labels_num[i] = i
    #     accuracy, precision, recall, f1_score = random_forest.main(training_set, test_set, num_trees, attr_type, data_labels_num, 10)
    #     # print(f"RF: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     rf_accuracies.append(accuracy)
    #     rf_precisions.append(precision)
    #     rf_recalls.append(recall)
    #     rf_f1_scores.append(f1_score)

    # print("MNIST Results:")
    # print(f"\tKNN: Accuracy: {sum(knn_accuracies) / num_folds}")
    # print(f"\tKNN: Precision: {sum(knn_precisions) / num_folds}")
    # print(f"\tKNN: Recall: {sum(knn_recalls) / num_folds}")
    # print(f"\tKNN: F1-Score: {sum(knn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tNN: Accuracy: {sum(nn_accuracies) / num_folds}")
    # print(f"\tNN: Precision: {sum(nn_precisions) / num_folds}")
    # print(f"\tNN: Recall: {sum(nn_recalls) / num_folds}")
    # print(f"\tNN: F1-Score: {sum(nn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tRF: Accuracy: {sum(rf_accuracies) / num_folds}")
    # print(f"\tRF: Precision: {sum(rf_precisions) / num_folds}")
    # print(f"\tRF: Recall: {sum(rf_recalls) / num_folds}")
    # print(f"\tRF: F1-Score: {sum(rf_f1_scores) / num_folds}")



    # digits stored as rows of 64 numbers
    # originally an 8x8 grayscale pixel array, which was flattened
    # I guess values 0->15 indicate white->black for each pixel
    # going to wait to write code to load this until we see how our NN turns out
def test_loans(num_folds: int, num_trees: int)-> None:
    k_folds, attr_type, attr_labels = misc.k_folds_gen(num_folds, os.path.join("The_Loan_Eligibility_Prediction_Dataset", "loan.csv"), True)

    # TODO: tune hyper-parameters, add random forest code
    num_neighbors = 10 # arbitrary, need to tune!
    knn_accuracies = []
    knn_precisions = []
    knn_recalls = []
    knn_f1_scores = []

    nn_accuracies = []
    nn_precisions = []
    nn_recalls = []
    nn_f1_scores = []

    rf_accuracies = []
    rf_precisions = []
    rf_recalls = []
    rf_f1_scores = []

    print(f"Beginning test on the loan eligibility dataset...")
    for k in range(num_folds):
        test_set = k_folds[k]
        training_set = []
        for i in range(1, len(k_folds)):
            if i != k:
                training_set += k_folds[i]

        if k == 0:
            # misc.learning_curve_knn(training_set, test_set, num_neighbors, 2)
            misc.learning_curve_nn(0, [11,15,15,15,2], training_set, test_set, 2)
            training_set.insert(0, attr_labels)
            data_labels_num = deepcopy(attr_labels)
            for i in range(len(data_labels_num) - 1):
                data_labels_num[i] = i
            # misc.learning_curve_rf(training_set, test_set, attr_type, data_labels_num, 2)
        
    #     # print(f"{k=}:")
    #     # print(f"\tTesting KNN:")
    #     accuracy, precision, recall, f1_score = knn.knn_test(training_set, test_set, num_neighbors, 2)
    #     #print(f"KNN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     knn_accuracies.append(accuracy)
    #     knn_precisions.append(precision)
    #     knn_recalls.append(recall)
    #     knn_f1_scores.append(f1_score)

    #     # print(f"\tTesting NN:")
    #     accuracy, precision, recall, f1_score = neural_net.main(0, [11,15,15,15,2], np.array(training_set), np.array(test_set), 2)
    #     #print(f"NN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     nn_accuracies.append(accuracy)
    #     nn_precisions.append(precision)
    #     nn_recalls.append(recall)
    #     nn_f1_scores.append(f1_score)
        
    #     # print(f"\tTesting RF:")
    #     # slap the labels back onto the top of the k_folds list of lists
    #     # TODO: ideally get rid of this
    #     training_set.insert(0, attr_labels)
    #     data_labels_num = deepcopy(attr_labels)
    #     for i in range(len(data_labels_num) - 1):
    #         data_labels_num[i] = i
    #     accuracy, precision, recall, f1_score = random_forest.main(training_set, test_set, num_trees, attr_type, data_labels_num, 2)
    #     # print(f"RF: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     rf_accuracies.append(accuracy)
    #     rf_precisions.append(precision)
    #     rf_recalls.append(recall)
    #     rf_f1_scores.append(f1_score)

    # print("Loan Eligibility Results:")
    # print(f"\tKNN: Accuracy: {sum(knn_accuracies) / num_folds}")
    # print(f"\tKNN: Precision: {sum(knn_precisions) / num_folds}")
    # print(f"\tKNN: Recall: {sum(knn_recalls) / num_folds}")
    # print(f"\tKNN: F1-Score: {sum(knn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tNN: Accuracy: {sum(nn_accuracies) / num_folds}")
    # print(f"\tNN: Precision: {sum(nn_precisions) / num_folds}")
    # print(f"\tNN: Recall: {sum(nn_recalls) / num_folds}")
    # print(f"\tNN: F1-Score: {sum(nn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tRF: Accuracy: {sum(rf_accuracies) / num_folds}")
    # print(f"\tRF: Precision: {sum(rf_precisions) / num_folds}")
    # print(f"\tRF: Recall: {sum(rf_recalls) / num_folds}")
    # print(f"\tRF: F1-Score: {sum(rf_f1_scores) / num_folds}")


def test_parkinsons(num_folds: int, num_trees: int)-> None:
    k_folds, attr_type, attr_labels = misc.k_folds_gen(num_folds, os.path.join("The_Oxford_Parkinson's_Disease_Detection_Dataset", "parkinsons.csv"), True)

    num_neighbors = 15 # arbitrary, need to tune!
    knn_accuracies = []
    knn_precisions = []
    knn_recalls = []
    knn_f1_scores = []

    nn_accuracies = []
    nn_precisions = []
    nn_recalls = []
    nn_f1_scores = []

    rf_accuracies = []
    rf_precisions = []
    rf_recalls = []
    rf_f1_scores = []

    print(f"Beginning test on the parkinson's dataset...")
    for k in range(num_folds):
        test_set = k_folds[k]
        training_set = []
        for i in range(1, len(k_folds)):
            if i != k:
                training_set += k_folds[i]

        if k == 0:
            # misc.learning_curve_knn(training_set, test_set, num_neighbors, 2)
            misc.learning_curve_nn(0, [22,15,15,15,2], training_set, test_set, 2)
            training_set.insert(0, attr_labels)
            data_labels_num = deepcopy(attr_labels)
            for i in range(len(data_labels_num) - 1):
                data_labels_num[i] = i
            # misc.learning_curve_rf(training_set, test_set, attr_type, data_labels_num, 2)
        
    #     # print(f"{k=}:")
    #     # print(f"\tTesting KNN:")
    #     accuracy, precision, recall, f1_score = knn.knn_test(training_set, test_set, num_neighbors, 2)
    #     #print(f"KNN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     knn_accuracies.append(accuracy)
    #     knn_precisions.append(precision)
    #     knn_recalls.append(recall)
    #     knn_f1_scores.append(f1_score)

    #     # print(f"\tTesting NN:")
    #     accuracy, precision, recall, f1_score = neural_net.main(0, [22,15,15,15,2], np.array(training_set), np.array(test_set), 2)
    #     #print(f"NN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     nn_accuracies.append(accuracy)
    #     nn_precisions.append(precision)
    #     nn_recalls.append(recall)
    #     nn_f1_scores.append(f1_score)

    #     # print(f"\tTesting RF:")
    #     # slap the labels back onto the top of the k_folds list of lists
    #     # TODO: ideally get rid of this
    #     training_set.insert(0, attr_labels)
    #     data_labels_num = deepcopy(attr_labels)
    #     for i in range(len(data_labels_num) - 1):
    #         data_labels_num[i] = i
    #     accuracy, precision, recall, f1_score = random_forest.main(training_set, test_set, num_trees, attr_type, data_labels_num, 2)
    #     # print(f"RF: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     rf_accuracies.append(accuracy)
    #     rf_precisions.append(precision)
    #     rf_recalls.append(recall)
    #     rf_f1_scores.append(f1_score)

    # print("Parkinson's Results:")
    # print(f"\tKNN: Accuracy: {sum(knn_accuracies) / num_folds}")
    # print(f"\tKNN: Precision: {sum(knn_precisions) / num_folds}")
    # print(f"\tKNN: Recall: {sum(knn_recalls) / num_folds}")
    # print(f"\tKNN: F1-Score: {sum(knn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tNN: Accuracy: {sum(nn_accuracies) / num_folds}")
    # print(f"\tNN: Precision: {sum(nn_precisions) / num_folds}")
    # print(f"\tNN: Recall: {sum(nn_recalls) / num_folds}")
    # print(f"\tNN: F1-Score: {sum(nn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tRF: Accuracy: {sum(rf_accuracies) / num_folds}")
    # print(f"\tRF: Precision: {sum(rf_precisions) / num_folds}")
    # print(f"\tRF: Recall: {sum(rf_recalls) / num_folds}")
    # print(f"\tRF: F1-Score: {sum(rf_f1_scores) / num_folds}")
    

def test_titanic(num_folds: int, num_trees: int)-> None:
    k_folds, attr_type, attr_labels = misc.k_folds_gen(num_folds, os.path.join("The_Titanic_Dataset", "titanic.csv"), True)

    num_neighbors = 8 # arbitrary, need to tune!
    knn_accuracies = []
    knn_precisions = []
    knn_recalls = []
    knn_f1_scores = []

    nn_accuracies = []
    nn_precisions = []
    nn_recalls = []
    nn_f1_scores = []

    rf_accuracies = []
    rf_precisions = []
    rf_recalls = []
    rf_f1_scores = []

    print(f"Beginning test on the Titanic dataset...")
    for k in range(num_folds):
        test_set = k_folds[k]
        training_set = []
        for i in range(1, len(k_folds)):
            if i != k:
                training_set += k_folds[i]

        if k == 0:
            # misc.learning_curve_knn(training_set, test_set, num_neighbors, 2)
            misc.learning_curve_nn(0, [6,11,11,2], training_set, test_set, 2)
            training_set.insert(0, attr_labels)
            data_labels_num = deepcopy(attr_labels)
            for i in range(len(data_labels_num) - 1):
                data_labels_num[i] = i
            # misc.learning_curve_rf(training_set, test_set, attr_type, data_labels_num, 2)

        
    #     # print(f"{k=}:")
    #     # print(f"\tTesting KNN:")
    #     accuracy, precision, recall, f1_score = knn.knn_test(training_set, test_set, num_neighbors, 2)
    #     #print(f"KNN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     knn_accuracies.append(accuracy)
    #     knn_precisions.append(precision)
    #     knn_recalls.append(recall)
    #     knn_f1_scores.append(f1_score)

    #     # print(f"\tTesting NN:")
    #     accuracy, precision, recall, f1_score = neural_net.main(0, [6,11,11,2], np.array(training_set), np.array(test_set), 2)
    #     #print(f"NN: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     nn_accuracies.append(accuracy)
    #     nn_precisions.append(precision)
    #     nn_recalls.append(recall)
    #     nn_f1_scores.append(f1_score)

    #     # print(f"\tTesting RF:")
    #     # slap the labels back onto the top of the k_folds list of lists
    #     # TODO: ideally get rid of this
    #     training_set.insert(0, attr_labels)
    #     data_labels_num = deepcopy(attr_labels)
    #     for i in range(len(data_labels_num) - 1):
    #         data_labels_num[i] = i
    #     accuracy, precision, recall, f1_score = random_forest.main(training_set, test_set, num_trees, attr_type, data_labels_num, 2)
    #     # print(f"RF: {accuracy=}, {precision=}, {recall=}, {f1_score=}")
    #     rf_accuracies.append(accuracy)
    #     rf_precisions.append(precision)
    #     rf_recalls.append(recall)
    #     rf_f1_scores.append(f1_score)

    # print("Titanic Results:")
    # print(f"\tKNN: Accuracy: {sum(knn_accuracies) / num_folds}")
    # print(f"\tKNN: Precision: {sum(knn_precisions) / num_folds}")
    # print(f"\tKNN: Recall: {sum(knn_recalls) / num_folds}")
    # print(f"\tKNN: F1-Score: {sum(knn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tNN: Accuracy: {sum(nn_accuracies) / num_folds}")
    # print(f"\tNN: Precision: {sum(nn_precisions) / num_folds}")
    # print(f"\tNN: Recall: {sum(nn_recalls) / num_folds}")
    # print(f"\tNN: F1-Score: {sum(nn_f1_scores) / num_folds}")
    # print(f"\n")
    # print(f"\tRF: Accuracy: {sum(rf_accuracies) / num_folds}")
    # print(f"\tRF: Precision: {sum(rf_precisions) / num_folds}")
    # print(f"\tRF: Recall: {sum(rf_recalls) / num_folds}")
    # print(f"\tRF: F1-Score: {sum(rf_f1_scores) / num_folds}")

def main():
    
    # test_loans(10, 35)
    # test_titanic(10, 35)
    # test_parkinsons(10, 15)
    test_digits(10, 35)
    # test_wine(10, 10)
    # test_congress(10, 10)

if __name__ == "__main__":
    main()