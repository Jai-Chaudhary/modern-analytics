#TITANIC
import pandas as pd
import numpy as np
import csv as csv
from sklearn import linear_model

def preprocess_data(filename):
        # Data cleanup
    # TRAIN DATA
    data_frame = pd.read_csv(filename, header=0)        # Load the train file into a dataframe

    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.

    # female = 0, Male = 1
    data_frame['Gender'] = data_frame['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(data_frame.Embarked[ data_frame.Embarked.isnull() ]) > 0:
        data_frame.Embarked[ data_frame.Embarked.isnull() ] = data_frame.Embarked.dropna().mode().values

    Ports = list(enumerate(np.unique(data_frame['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    data_frame.Embarked = data_frame.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = data_frame['Age'].dropna().median()
    if len(data_frame.Age[ data_frame.Age.isnull() ]) > 0:
        data_frame.loc[ (data_frame.Age.isnull()), 'Age'] = median_age

    # All the missing Fares -> assume median of their respective class
    if len(data_frame.Fare[ data_frame.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = data_frame[ data_frame.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            data_frame.loc[ (data_frame.Fare.isnull()) & (data_frame.Pclass == f+1 ), 'Fare'] = median_fare[f]

    # Collect the test data's PassengerIds before dropping it
    ids = data_frame['PassengerId'].values
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    data_frame = data_frame.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
    print data_frame
    return ids, data_frame.values



if __name__ == '__main__':
    train_ids, train_data = preprocess_data('train.csv')
    test_ids, test_data = preprocess_data('test.csv')
    clf = linear_model.LogisticRegression()
    clf.fit(train_data[0::,1::], train_data[0::,0])
    survived_pred = clf.predict(np.matrix(test_data)).astype(int)

    predictions_file = open("test_prediction.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(test_ids, survived_pred))
    predictions_file.close()
    print 'Done.'