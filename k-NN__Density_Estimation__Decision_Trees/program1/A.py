__author__ = 'Jai Chaudhary'

from sklearn.neighbors.kde import KernelDensity
import numpy as np
import argparse
import code.utils as utils
from config import TRIP_DATA_1,TRIP_DATA_2,EXAMPLE_DATA,F_FIELDS,S_FIELDS
import matplotlib.pyplot as plt
import matplotlib
from sklearn.grid_search import GridSearchCV

DEFAULT_DATASET = 'train_data.csv'
DATASETS = [ DEFAULT_DATASET]

DEFAULT_MODEL = 'gaussian-kernel'
MODELS = [ DEFAULT_MODEL]


def EstimateDensity(dataset, model):
	example_data = np.array([row[-2:] for row in utils.load_csv_lazy(EXAMPLE_DATA, S_FIELDS,F_FIELDS,row_filter=utils.distance_filter)], dtype = float)
	print example_data[:,-2:]
	example_data = example_data[:5000,-2:]
	# plt.scatter(example_data[:,-2:-1], example_data[:,-1])
	params = {'bandwidth': np.logspace(-10, 1, 1)}
	grid = GridSearchCV(KernelDensity(kernel='gaussian', metric="haversine"), params)
	grid.fit(example_data[:,-2:])
	kde = grid.best_estimator_
	# kde = KernelDensity(kernel='gaussian', metric="haversine", bandwidth=0.0002)
	# kde.fit(example_data[:,-2:])
	# kernel = stats.gaussian_kde(example_data[:,-2:])
	
	xmin = float(min(example_data[:,-2:-1])[0])
	xmax = float(max(example_data[:,-2:-1])[0])
	ymin = float(min(example_data[:,-1:])[0])
	ymax = float(max(example_data[:,-1:])[0])
	
	# print xmin,xmax,ymin,ymax
	# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	x = np.linspace(xmin, xmax, 100)
	y = np.linspace(ymin, ymax, 100)
	# print x, y
	X, Y = np.meshgrid(x, y)
	# print X, Y
	positions = np.vstack([X.ravel(), Y.ravel()]).T
	print positions
	Z = np.reshape(kde.score_samples(positions), X.shape)
	levels = np.logspace(Z.min(), Z.max(), 10)

	CS = plt.contourf(X, Y, Z, cmap=plt.cm.bone)
	CS2 = plt.contour(CS, levels=levels,
                        colors = 'r',
                        hold='on', norm=matplotlib.colors.LogNorm())
	plt.scatter(example_data[:,-2:-1], example_data[:,-1])
	plt.show()







def main():
	parser = argparse.ArgumentParser( description = 'Density Estimation and Plot' )
	parser.add_argument( 'dataset'     , nargs = '?', type = str, default = False, help = 'Dataset File' )
	parser.add_argument( 'model'       , nargs = '?', type = str, default = DEFAULT_MODEL  , choices = MODELS  , help = 'Model type' )
	args = parser.parse_args()
	EstimateDensity( args.dataset, args.model )

if __name__ == '__main__':
	main()
