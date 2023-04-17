from copy import deepcopy
import csv
from random import randrange
from statistics import stdev
import matplotlib.pyplot as plt

# read in the data from the csv, normalize it, randomly split into training and test sets, return these sets
def prepare_data():
    # load in data
    data_set = [] # entire dataset, as read in from the csv
    training_set = [] # training set partition of the entire data set
    test_set = [] # testing partition of the entire data set
    num_attr = 0
    num_entries = 0

    with open('iris.csv') as raw_data_file:
        data_reader = csv.reader(raw_data_file)
        data_set = list(data_reader)
    
    # grab some general info about the data set...
    # see how many attributes are in the data set
    for item in data_set[0]:
        num_attr += 1
    # just assume the last item in a row is the label....
    num_attr -= 1
    num_entries = len(data_set)
    
    # sanity checks
    if num_attr <= 0 or num_entries == 0:
        print("Error: Bad data")
        return
    else:
        #print(f"Dataset has {num_entries} entries with {num_attr} attributes.")
        pass

    # I imagine this doesn't generalize to other data sets, but for this one we'll cast all of the attributes to floats
    for entry in range(num_entries):
        for attr in range(num_attr):
            data_set[entry][attr] = float(data_set[entry][attr])

    # split between training (80%) and testing data (20%)
    training_set_size = int(num_entries * 0.8)
    test_set_size = int(num_entries * 0.2)
    # deal with any potential rounding issues
    while (training_set_size + test_set_size) > num_entries:
        test_set_size -= 1
    while (training_set_size + test_set_size) < num_entries:
        training_set_size += 1

    # randomly select the entries to go into the training set, note this selection method also randomizes the order
    for _ in range(training_set_size):
        training_set.append(data_set.pop(randrange(num_entries)))
        num_entries -= 1

    test_set = deepcopy(data_set)

    # normalize the data attributes, minimum is zero so we just need to find the maximum value for each attribute
    for attr in range(num_attr):
        max_val = 0
        #max_val = max(training_set[:,attr]) # would like to vectorize this operation but I don't know python that well...
        for entry in range(training_set_size): # find the max value for the given attribute in the training data
            max_val = training_set[entry][attr] if (training_set[entry][attr] > max_val) else max_val
        for entry in range(training_set_size): # now scale the given attribute according to that attribute
            training_set[entry][attr] /= max_val
        for entry in range(test_set_size): # scale test data too
            test_set[entry][attr] /= max_val

    return training_set, test_set

def euclidean_dist(x_1, x_2):
    accum = 0
    for i in range(len(x_1)):
        accum += (x_1[i] - x_2[i])**2
    return accum**0.5

def knn_classify(neighbors, instance, k, distance_func):
    neighbors_local = deepcopy(neighbors) # make a copy of neighbors so we don't mess with the original list's contents
    
    for i in range(len(neighbors_local)):
        neighbors_local[i].append(distance_func(neighbors_local[i][:len(instance) - 1], instance[:len(instance) - 1]))
    
    neighbors_local.sort(key=lambda x: x[-1]) # sort neighbors according to the last entry (euclidean distance that was just appended on)

    votes = {} # dict to track the nearest neighbor's votes
    for i in range(k):
        if neighbors_local[i][-2] in votes: # if there's already been a vote for this class...
            votes[neighbors_local[i][-2]] += 1 # ...increment the count
        else:
            votes[neighbors_local[i][-2]] = 1 # ...otherwise add it to the dict

    return max(votes, key = votes.get)
            

#def main():
#    x_points = []
#    training_points = []
#    test_points = []
#    training_stdev = []
#    test_stdev = []
#    for k in range(1, 51 + 1, 2): # k = 1, 3, 5, ..., 51
#        print(f'k = {k}')
#        x_points.append(k)
#        training_set_acc = []
#        test_set_acc = []
#        for trial_num in range(100):
#            training_set, test_set = prepare_data()
#            num_correct = 0
#            for x in training_set:
#                if knn_classify(training_set, x, k, euclidean_dist) == x[-1]:
#                    num_correct += 1
#            training_set_acc.append(num_correct / len(training_set))

#            num_correct = 0
#            for x in test_set:
#                if knn_classify(training_set, x, k, euclidean_dist) == x[-1]:
#                    num_correct += 1
#            test_set_acc.append(num_correct / len(test_set))
#        print(f'        Training Set Accuracy: {sum(training_set_acc) / 100}, stdev: {stdev(training_set_acc)}')
#        print(f'        Test Set Accuracy: {sum(test_set_acc) / 100}, stdev: {stdev(test_set_acc)}')
#        training_points.append(sum(training_set_acc) / 100)
#        training_stdev.append(stdev(training_set_acc))
#        test_points.append(sum(test_set_acc) / 100)
#        test_stdev.append(stdev(test_set_acc))
#    training_plot = plt.figure(1)
#    plt.xlabel('k')
#    plt.ylabel('Accuracy (%)')
#    plt.title("Training Set Accuracy")
#    plt.errorbar(x_points, training_points, yerr=training_stdev,linestyle='-',fmt="o", capsize=5)
#    training_plot.show()

#    test_plot = plt.figure(2)
#    plt.xlabel('k')
#    plt.ylabel('Accuracy (%)')
#    plt.title("Test Set Accuracy")
#    plt.errorbar(x_points, test_points, yerr=test_stdev,linestyle='-',fmt="o", capsize=5)
#    test_plot.show()

#    input()
#    #throw_away = input("Enter anything to end the program...") # graphs immediately dissapear when the program ends

def main():
    x_points = []
    training_points = []
    test_points = []
    training_stdev = []
    test_stdev = []
    training_set_acc = []
    test_set_acc = []
    final_testing = []
    final_training = []
    x_points = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51]

    for k in range(100):
        final_testing.append(0)
        final_training.append(0)

    for run_num in range(20):
        training_set, test_set = prepare_data()
        for k in range(1, 51 + 1, 2):
            num_correct = 0
            for x in training_set:
                if knn_classify(deepcopy(training_set), x, k, euclidean_dist) == x[-1]:
                    num_correct += 1
            print(f'run num: {run_num}, k: {k}, training acc: {num_correct / len(training_set)}')
            final_training[k] += num_correct / len(training_set)

            num_correct = 0
            for x in test_set:
                if knn_classify(deepcopy(training_set), x, k, euclidean_dist) == x[-1]:
                    num_correct += 1
            print(f'run num: {run_num}, k: {k}, testing acc: {num_correct / len(test_set)}')
            final_testing[k] += num_correct / len(test_set)
    
    final_training2 = []
    final_testing2 = []

    for k in range(1, 51 + 1, 2):
        final_testing[k] /= 20
        final_training[k] /= 20
        final_training2.append(final_training[k])
        final_testing2.append(final_testing[k])
           
    training_plot = plt.figure(1)
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.title("Training Set Accuracy")
    plt.scatter(x_points, final_training2)
    training_plot.show()

    testing_plot = plt.figure(1)
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.title("Training Set Accuracy")
    plt.scatter(x_points, final_testing2)
    testing_plot.show()
    
    #print("final training:")
    #print(final_training)
    #print("final testing:")
    #print(final_testing)

    ##training_points.append(sum(training_set_acc) / 100)
    ##training_stdev.append(stdev(training_set_acc))
    ##test_points.append(sum(test_set_acc) / 100)
    ##test_stdev.append(stdev(test_set_acc))
    #training_plot = plt.figure(1)
    #plt.xlabel('k')
    #plt.ylabel('Accuracy (%)')
    #plt.title("Training Set Accuracy")
    #plt.scatter(x_points, final_training)
    ##plt.errorbar(x_points, training_points, yerr=training_stdev,linestyle='-',fmt="o", capsize=5)
    ##plt.errorbar(x_points, final_training,fmt="o", capsize=5)
    #training_plot.show()

    #test_plot = plt.figure(2)
    #plt.xlabel('k')
    #plt.ylabel('Accuracy (%)')
    #plt.title("Test Set Accuracy")
    #plt.scatter(x_points, final_testing)
    ##plt.errorbar(x_points, test_points, yerr=test_stdev,linestyle='-',fmt="o", capsize=5)
    ##plt.errorbar(x_points, final_test, linestyle='-',fmt="o", capsize=5)
    #test_plot.show()

    input()
#    #throw_away = input("Enter anything to end the program...") # graphs immediately dissapear when the program ends

if __name__ == '__main__':
    main()