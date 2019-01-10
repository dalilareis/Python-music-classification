import sys
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


#------------------------------------------------------LOAD FILES, PRE-PROCESS AND NORMALIZE-------------------------------------------------------------

def loadFeatures_NoSplit(filepath='features_final.csv'):
	features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
	X = features.iloc[:, 0:518].values
	y = features.iloc[:, 518].values

	enc = LabelEncoder()
	y = enc.fit_transform(y)
	genre_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_, ))
	X, y = shuffle(X, y, random_state=42)

	return X, y, genre_mapping

def normalize(X, scaler='quantileNormal'):
	if scaler=='standard':
		X = StandardScaler().fit_transform(X)
	elif scaler=='minMax':
		X = MinMaxScaler().fit_transform(X)
	elif scaler=='maxAbs':
		X = MaxAbsScaler().fit_transform(X)
	elif scaler=='robust':
		X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
	elif scaler=='quantileUniform':
		X = QuantileTransformer(output_distribution='uniform').fit_transform(X)
	elif scaler=='quantileNormal':
		X = QuantileTransformer(output_distribution='normal').fit_transform(X)
	elif scaler=='L2Norm':
		X = Normalizer().fit_transform(X)
	else:
		raise ValueError('Scaler is not defined: %s' % scaler)

	return X

#------------------------------------FEATURE SELECTION (Comparison PCA, NMF, kBest)------------------------------

def test_selectors(filepath='features_final.csv', scaler='minMax'):
	from sklearn.externals.joblib import Memory
	from tempfile import mkdtemp
	from shutil import rmtree

	X, y, genre_mapping = loadFeatures_NoSplit(filepath)
	X = normalize(X, scaler)

	pipe = Pipeline([
	    ('reduce_dim', PCA()),
	    ('classify', LinearSVC())
	])

	N_FEATURES_OPTIONS = [2, 10, 20, 40, 60, 120]
	C_OPTIONS = [1, 10]
	param_grid = [
	    {
	        'reduce_dim': [PCA(iterated_power=7), NMF()],
	        'reduce_dim__n_components': N_FEATURES_OPTIONS,
	        'classify__C': C_OPTIONS
	    },
	    {
	        'reduce_dim': [SelectKBest(chi2)],
	        'reduce_dim__k': N_FEATURES_OPTIONS,
	        'classify__C': C_OPTIONS
	    },
	]
	reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(cachedir=cachedir, verbose=10)
	cached_pipe = Pipeline([('reduce_dim', PCA()), ('classify', LinearSVC())], memory=memory)

	# This time, a cached pipeline will be used within the grid search
	grid = GridSearchCV(cached_pipe, cv=2, n_jobs=1, param_grid=param_grid)
	#grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
	grid.fit(X, y)

	# Delete the temporary cache before exiting
	rmtree(cachedir)

	mean_scores = np.array(grid.cv_results_['mean_test_score'])
	# scores are in the order of param_grid iteration, which is alphabetical
	mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
	# select score for best C
	mean_scores = mean_scores.max(axis=0)
	bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
	               (len(reducer_labels) + 1) + .5)

	plt.figure()
	COLORS = 'bgrcmyk'
	for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
	    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

	plt.title("Comparing feature reduction techniques")
	plt.xlabel('Reduced number of features')
	plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
	plt.ylabel('Classification accuracy')
	plt.ylim((0, 1))
	plt.legend(loc='upper left')
	plt.show()	

#----------------------------------------------------------CLASSIFICATION MODELS----------------------------------------------------------------

#+++++++++++++++++++++Search for Best Hyperparameters for each Classification Model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def report_params(results, n_top=2):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def model_params(model):
    if model=='RF':
        param_dist = {"n_estimators": [20, 50, 100],
              "max_depth": [3, None],
              "max_features": stats.randint(1, 7),
              "min_samples_split": stats.randint(2, 11),
              "min_samples_leaf": stats.randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    elif model=='SVM':
        param_dist = {"C": stats.expon(scale=100),
              "kernel": ["linear", "poly", "rbf", "sigmoid"],
              "gamma": stats.expon(scale=.1)}
    elif model=='LR':
        param_dist = {"penalty": ["l2"],
              "C": stats.expon(scale=100),
              "solver": ["newton-cg", "sag", "lbfgs"],
              "multi_class": ["multinomial"],
              "warm_start": [True, False]}
    elif model=='kNN':
        param_dist = {"n_neighbors": stats.randint(5, 30),
              "algorithm": ["ball_tree", "kd_tree", "brute"],
              "leaf_size": stats.randint(5, 50)}
    elif model=='MLP':
        param_dist = {"activation": ["identity", "logistic", "tanh", "relu"],
              "solver": ["lbfgs", "sgd", "adam"],
              "alpha": stats.expon(scale=.1),
              "batch_size": stats.randint(50, 500)}
    elif model=='SGD':
        param_dist = {"loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
              "penalty": ["none", "l2", "l1", "elasticnet"],
              "alpha": stats.expon(scale=.1),
              "max_iter": [5, 50]}
    elif model=='extraTrees':
         param_dist = {"n_estimators": [10, 20, 50, 100],
              "max_depth": [3, None],
              "max_features": stats.randint(1, 7),
              "min_samples_split": stats.randint(2, 11),
              "min_samples_leaf": stats.randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    else:
    	raise ValueError('Hyperparameters not specified for {} model.'.format(model))

    return param_dist

def random_search(X, y, model_name, model, n_iter=50):
    from time import time
    param_dist = model_params(model_name)
    randomSearch = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, n_jobs=2, cv=3)
    start = time()
    randomSearch.fit(X, y)
    print("***** RandomizedSearchCV for model %s took %.2f seconds for %d candidates parameter settings *****" % (model_name, (time() - start), n_iter))
    report_params(randomSearch.cv_results_)

def tune_params(model_name, filepath='features_final.csv'):
	X, y, genre_mapping = loadFeatures_NoSplit(filepath)

	X = normalize(X, scaler='quantileNormal')
	X = LDA().fit(X, y).transform(X)
	X, y = shuffle(X, y, random_state=42)

	if model_name == 'RF':
		model = RandomForestClassifier()
	elif model_name == 'SVM':
		model = SVC()
	elif model_name == 'LR':
		model = LogisticRegression()
	elif model_name == 'kNN':
		model = KNeighborsClassifier()
	elif model_name == 'MLP':
		model = MLPClassifier()
	elif model_name == 'SGD':
		model = SGDClassifier()
	elif model_name == 'extraTrees':
		model = ExtraTreesClassifier()
	else:
		raise ValueError('Model not recognized: %s' % model)

	random_search(X, y, model_name, model)

#--------------------------------------------------------------------MAIN-------------------------------------------------------------------------

if __name__ == "__main__":

	scalers = ['standard', 'minMax', 'maxAbs', 'robust', 'quantileUniform', 'quantileNormal', 'L2Norm']
	models = ['RF', 'LR', 'SVM', 'kNN', 'SGD', 'MLP', 'extraTrees']

#------Search best parameters for a given model-----------
	#use command: python testing.py params "chosen model" e.g. python testing.py params RF
	#test_selectors(scaler='minMax')	
	if sys.argv[1] == 'params':
		tune_params(sys.argv[2])

#-----Compare 3 different reducers (PCA, NMF, kBest) using Linear SVM Classifier-------
	#use command: python testing.py compare "chosen scaler" e.g. use command: python testing.py compare robust
	#tune_params('RF')
	elif sys.argv[1] == 'compare':
		test_selectors(sys.argv[2])
