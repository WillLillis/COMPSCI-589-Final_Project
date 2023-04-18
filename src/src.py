# TODO:
    # - Clean up existing code
        # - Remove dependence on data labels being in the first row, just consolidate to an argument
    # - Fill in functions below to load respective data sets
    # - Then write testing functions that will train a specified model 
    # on the loaded data and evaluate it using the metrics of interest 
        # - k-fold stratified cross validation with k=10
        # - Accuracy and F1 score
import misc
import os

def test_digits():
    pass
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
    test_parkinsons()

if __name__ == "__main__":
    main()