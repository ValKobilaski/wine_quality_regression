import numpy as np

def load_data(file_name='winequality-red.csv', verbose = False):
    data = []
    split_ratio = 0.8

    with open(file_name,'r') as input:
        for line in input:
            data.append(line.split(','))

    np_data = np.matrix(data[1:])
    mask = np.random.rand(np_data.shape[0]) <= split_ratio
    train_data = np_data[mask]
    test_data = np_data[~mask]

    x_train = train_data[:,:-1] 
    y_train = train_data[:,-1]
    x_test = test_data[:,:-1]
    y_test = test_data[:,-1]

    return x_train, y_train, x_test, y_test
    

load_data()