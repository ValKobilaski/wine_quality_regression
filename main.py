import numpy as np

def load_data(file_name='winequality-red.csv', verbose = False, split_ratio = 0.8):
    """Loads dataset in CSVinto training and test sets

    Keyword arguments:
    file_name -- path tot dataset file (default 'winequality-red.csv)
    verbose -- specifies whether metadata should be printed (default False)
    split_ratio -- the ratio of training data to test data (default 0.8)
    """

    data = []

    #Read file into data (2D array)
    with open(file_name,'r') as input:
        for line in input:
            data.append(line.split(','))

    #Convert to numpy array
    np_data = np.matrix(data[1:])

    #Split data into train and test
    mask = np.random.rand(np_data.shape[0]) <= split_ratio
    train_data = np_data[mask]
    test_data = np_data[~mask]

    # seperate train and test into X and Y
    x_train = train_data[:,:-1] 
    y_train = train_data[:,-1]
    x_test = test_data[:,:-1]
    y_test = test_data[:,-1]

    if (verbose):
        #print metadata on dataset
        print('Loading dataset:', file_name)
        print('*****************************')
        print('Number of data points:', np_data.shape[0])
        print('Number of input variables:', np_data.shape[1])
        print('Training data points:', x_train.shape[0])
        print('Test set data points:', x_test.shape[0])

    return x_train, y_train, x_test, y_test
    

load_data(verbose = True)