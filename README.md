# COMPSCI-589-Final_Project

## Instructions on Running Code

The main code is located in src/src.py and all relevant helper functions are located in src/misc.py. To run the program simply run

`python3 src.py`

At the bottom of src.py, there are 4 functions that we use: `test_loans()`, `test_digits()`, `test_parkinsons()`, and `test_titanic()`. The parameters passed in to these funtions are num_folds and num_trees, respectively, and other fine-tuning of algorithms is done inside the methods themselves. 
The various ``test_`` functions run stratified k-folds cross validation on the given dataset, using Neural Networks, Random Forests, and k-Nearest Neighbors. After evaluating all k folds, it prints out the averaged results.

The source code for the implementation of each algorithm is contained within the corresponding sub-directories within ``src``. 

For creating the learning curves, the code is commented out in each respective method and implementation of the testing code can be found in misc.py.