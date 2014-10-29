__author__ = 'Jai Chaudhary'

from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import argparse
import code.utils as utils
from config import TRIP_DATA_1,TRAIN_DATA,F_FIELDS,S_FIELDS
from datetime import datetime

normalize_u = None
normalize_s = None

def set_normalize_params(records):
    global normalize_u
    global normalize_s
    normalize_u = np.mean(records, axis=0) 
    normalize_s = np.std(records, axis=0)


def normalize(records):
    return np.divide(np.subtract(records, normalize_u * np.ones(records.shape, dtype=np.float)), normalize_s * np.ones(records.shape, dtype=np.float)) 

def datestring_to_seconds_from_midnight(dateStr):
    datetimeObj = datetime.strptime(dateStr, "%Y-%m-%d %H:%M:%S")
    return datetimeObj.hour * 3600 + datetimeObj.minute * 60 + datetimeObj.second

def oneNN(trainOn, testOn):
    trainX = np.array([[datestring_to_seconds_from_midnight(row[0])] + row[-5:] for row in utils.load_csv_lazy(TRAIN_DATA, S_FIELDS,F_FIELDS,row_filter=utils.distance_filter)], dtype = float)
    set_normalize_params(trainX)
    trainX = normalize(trainX)
    
    trainY = np.array([row[2] for row in utils.load_csv_lazy(TRAIN_DATA, S_FIELDS,F_FIELDS,row_filter=utils.distance_filter)], dtype = float)

    nbrs = KNeighborsRegressor(n_neighbors=1, algorithm='ball_tree').fit(trainX[:trainOn], trainY[:trainOn])
    print "Train Complete"
    
    testX = np.array([[datestring_to_seconds_from_midnight(row[0])] + row[-5:] for row in utils.load_csv_lazy(TRIP_DATA_1, S_FIELDS,F_FIELDS,row_filter=utils.distance_filter)], dtype = float)
    testX = normalize(testX)
    
    testY = np.array([row[2] for row in utils.load_csv_lazy(TRIP_DATA_1, S_FIELDS,F_FIELDS,row_filter=utils.distance_filter)], dtype = float)
    
    print utils.metrics(nbrs, testX[:testOn], testY[:testOn])

def main():
    parser = argparse.ArgumentParser( description = '1-Nearest Neighbour to predict trip_time' )
    parser.add_argument( 'trainOn' , nargs = '?', type = int, default = 100, help = 'Number of data points to train on' )
    parser.add_argument( 'testOn' , nargs = '?', type = int, default = 100, help = 'Number of data points to test on' )
    args = parser.parse_args()
    oneNN( args.trainOn, args.testOn )

if __name__ == '__main__':
    main()
