import math
import numpy as numpy
# import pandas as pd
from pprint import pprint
from matplotlib import dates as dates
from datetime import datetime as datetime
import matplotlib.pyplot as plt
from sklearn import linear_model

def get_distance(lat1, long1, lat2, long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    # MODIFIED TO return distance in miles
    return arc*3960.0


def outlier_filter_percentile(x):
    if x.shape == (len(x),) :
        q75, q25 = numpy.percentile(x, [75 ,25])
        return [data_point for data_point in x if q25 < data_point < q75 ]
    else:
        unfiltered_x = x  
        q75, q25  = numpy.percentile(x, [75 ,25], axis = 0)
        for i in xrange(len(x[0])):
            filtered_x = [data_point for data_point in unfiltered_x if (q25[i] < data_point[i] < q75[i]) ]
            unfiltered_x = filtered_x
        return numpy.array(filtered_x)

def outlier_filter_std_dev(x):
    m = 2
    u = numpy.mean(x, axis=0)
    s = numpy.std(x, axis=0)
    if x.shape == (len(x),) :
        q75, q25 = numpy.percentile(x, [75 ,25])
        return [data_point for data_point in x if (u - m * s < data_point < u + m * s) ]
    else:
        unfiltered_x = x  
        for i in xrange(len(x[0])):
            filtered_x = [data_point for data_point in unfiltered_x if (u[i] - m * s[i] < data_point[i] < u[i] + m * s[i]) ]
            unfiltered_x = filtered_x
        return numpy.array(filtered_x)


def scatter_plots(data_features):
    (data_points_count,attributes_count) = data_features.shape
    figure, subplots = plt.subplots(attributes_count, attributes_count)
    figure.suptitle('Taxi Data')
    label_names = ['pickup_time', 'trip_distance', 'shortest_distances','trip_time']
    transpose_data_feature = [[row[i] for row in data_features] for i in range(attributes_count)]

    for x_axis in range(attributes_count):
        for y_axis in range(attributes_count):
            if x_axis != y_axis:
                subplot = subplots[x_axis][y_axis]
                subplot.scatter(transpose_data_feature[x_axis], transpose_data_feature[y_axis])
                subplot.set_xlabel(label_names[x_axis])
                subplot.set_ylabel(label_names[y_axis])
    plt.subplots_adjust(wspace=0.25, hspace=0.25,left=0.1,right=0.9, bottom=0.1)
    plt.show()

def test_train_distribute(data_features):
    test_data = []
    training_data = []
    for i in xrange(len(data_features)):
        if i%4 == 0:
            test_data.append(data_features[i])
        else:
            training_data.append(data_features[i])
    return numpy.array(test_data), numpy.array(training_data)

def linear_regression(training_input_features, training_output_var, test_input_features, test_output_var):
    clf = linear_model.LinearRegression()
    clf.fit(training_input_features, training_output_var)
    predicted_var = clf.predict(test_input_features)
    OLS_error = numpy.dot(predicted_var - test_output_var, predicted_var - test_output_var)
    orthogonal_error_vector = []
    for i in xrange(len(test_output_var)):
        orthogonal_error_vector.append(orthogonal_distance_of_prediction(test_input_features[i], predicted_var[i], test_output_var[i], clf.coef_))
    TLS_error = numpy.dot(orthogonal_error_vector, orthogonal_error_vector)
    correlation_coefficient = clf.score(test_input_features, test_output_var)
    print OLS_error, TLS_error, correlation_coefficient

def orthogonal_distance_of_prediction(date_point, data_point_y_predicted, data_point_y_orig, coefficients):
    # src: http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    normal_vector = numpy.append(coefficients,-1)
    normal_unit_vector = normal_vector / numpy.linalg.norm(normal_vector)
    point_on_line = numpy.append(date_point, data_point_y_predicted)
    point_to_measure_distance_from = numpy.append(date_point, data_point_y_orig)
    vector_from_orig_point_to_point_on_line = point_on_line - point_to_measure_distance_from
    projection_of_vector_from_orig_point_to_point_on_line_with_normal_vector = numpy.dot(vector_from_orig_point_to_point_on_line,normal_unit_vector) * normal_unit_vector
    vector_from_orig_point_to_orthogonal_point_on_line = vector_from_orig_point_to_point_on_line - projection_of_vector_from_orig_point_to_point_on_line_with_normal_vector
    orthogonal_distance = numpy.linalg.norm(vector_from_orig_point_to_orthogonal_point_on_line)
    return orthogonal_distance

def load():
    distances = []
    data_features = []
    trip_times = []
    error_count = 0
    for line in file('example_data.csv'):
        line = line.strip().split(',')
        trip_time,trip_distance,plong,plat,dlong,dlat=line[-6:]
        pickup_time = line[6]
        try:
            plong = float(plong)
            plat = float(plat)
            dlong = float(dlong)
            dlat = float(dlat)
            shortest_distances = get_distance(plat,plong,dlat,dlong)
            distances.append(shortest_distances)
            pickup_time = dates.date2num(datetime.strptime(pickup_time, "%Y-%m-%d %H:%M:%S"))
            trip_distance = float(trip_distance)
            data_features.append([pickup_time, trip_distance, shortest_distances, int(trip_time)])
        except:
            error_count += 1
            print error_count,plong,plat,dlong,dlat
    filtered_features = outlier_filter_percentile(numpy.array(data_features))
    scatter_plots(numpy.array(filtered_features))
    test_data, training_data = test_train_distribute(data_features)
    linear_regression(training_data[:,0:2], training_data[:,3], test_data[:,0:2], test_data[:,3])


if __name__ == '__main__':
    load()