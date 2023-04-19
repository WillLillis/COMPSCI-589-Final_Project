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
import os

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
    test_digits()

if __name__ == "__main__":
    main()