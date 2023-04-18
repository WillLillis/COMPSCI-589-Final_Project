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
    pass
def test_parkinsons():
    pass
def test_titanic():
    pass


def main():
    k_folds, attr_type, attr_labels = misc.k_folds_gen(10, os.path.join("The_Titanic_Dataset", "titanic.csv"))

    print(f"attr_type: {attr_type}")
    print(f"attr_labels: {attr_labels}")
    for k in range(len(k_folds)):
        print(f"Fold {k}:")
        print(k_folds[k])

if __name__ == "__main__":
    main()