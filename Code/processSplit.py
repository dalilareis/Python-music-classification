
import os
import multiprocessing
import warnings
import numpy as np
import itertools
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, classification_report, accuracy_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_predict, learning_curve
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


#------------------------------------------------------LOAD FILES, PRE-PROCESS AND NORMALIZE-------------------------------------------------------------

def load_split(filepath):
	#Load csv file
	features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2]) #index_col=0 uses 1st column as row names (starts counting at 2nd column)

	#Get labels from data and split data into train and test sets
	X = features.iloc[:, 0:518].values
	y = features.iloc[:, 518].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
	X_train, y_train = shuffle(X_train, y_train, random_state=42)

	#Transform genres to integer labels
	enc = LabelEncoder()
	y_train = enc.fit_transform(y_train)
	y_test = enc.transform(y_test)
	genre_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_, ))

	return (X_train, y_train), (X_test, y_test), genre_mapping

def loadFeatures(filepath):
	features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
	X = features.iloc[:, 0:518].values
	y = features.iloc[:, 518].values

	enc = LabelEncoder()
	y = enc.fit_transform(y)
	genre_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_, ))
	X, y = shuffle(X, y, random_state=42)

	return X, y, genre_mapping

def load_subsets(filepath, subset):
	features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2]) 
	y = features.iloc[:, 518].values
	enc = LabelEncoder()
	y = enc.fit_transform(y)
	genre_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_, ))

	if subset=='chroma_cens':
		X = features.iloc[:, 0:84].values
	elif subset=='chroma_cqt':
		X = features.iloc[:, 84:168].values
	elif subset=='chroma_stft':
		X = features.iloc[:, 168:252].values
	elif subset=='mfcc':
		X = features.iloc[:, 252:392].values
	elif subset=='rmse':
		X = features.iloc[:, 392:399].values
	elif subset=='spec_bandwidth':
		X = features.iloc[:, 399:406].values
	elif subset=='spec_centroid':
		X = features.iloc[:, 406:413].values
	elif subset=='spec_contrast':
		X = features.iloc[:, 413:462].values
	elif subset=='spec_rolloff':
		X = features.iloc[:, 462:469].values
	elif subset=='tonnetz':
		X = features.iloc[:, 469:511].values
	elif subset=='zcr':
		X = features.iloc[:, 511:518].values
	else:
		raise ValueError('Subset is not defined: %s' % subset)
	
	X, y = shuffle(X, y, random_state=42)

	return X, y, genre_mapping

def combine_subsets(filepath, listSubsets):
	listX = []
	listY = []

	for subset in listSubsets:
		X, y, genre_mapping = load_subsets(filepath, subset)
		listX.append(X)
		listY.append(y)

	X = np.concatenate((listX), axis=1)
	y = np.concatenate((listY), axis=1)
	#combo_names = ' + '.join(listSubsets)
	#setX = {combo_names: X, y}

	return X, y

def normalize(X_train, X_test, scaler='quantileNormal'):
	if scaler=='standard':
		scale = StandardScaler().fit(X_train)
		X_train = scale.transform(X_train)
		X_test = scale.transform(X_test)
	elif scaler=='minMax':
		X = MinMaxScaler().fit_transform(X)
	elif scaler=='maxAbs':
		X = MaxAbsScaler().fit_transform(X)
	elif scaler=='robust':
		X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
	elif scaler=='quantileUniform':
		X = QuantileTransformer(output_distribution='uniform').fit_transform(X)
	elif scaler=='quantileNormal':
		scale = QuantileTransformer(output_distribution='normal').fit(X_train)
		X_train = scale.transform(X_train)
		X_test = scale.transform(X_test)
	elif scaler=='L2Norm':
		X = Normalizer().fit_transform(X)
	else:
		raise ValueError('Scaler is not defined: %s' % scaler)

	return X_train, X_test
		
#---------------------------------------------------PRINCIPAL COMPONENT ANALYSIS - PCA-------------------------------------------------------

def test_pca(filepath, variance=0.75):
    (X_train, y_train), (X_test, y_test), genre_mapping = load_split(filepath)
    X_train = normalize(X_train)
    pca = PCA(variance)
    pca.fit_transform(X_train)
    selected = pca.n_components_
    print('Selected {} principal components from {} original features.'.format(selected, X_train.shape[1]))
    #95% variance --> 206 components 	90% --> 145		85% --> 109 	80% --> 82 		75% --> 63		60% --> 27
 
def apply_pca(train, test, variance):
	pca = PCA(variance)
	X_train = pca.fit_transform(train)
	X_test = pca.transform(test)
	selected = pca.n_components_
	print('Selected {} principal components from {} original features.'.format(selected, X_train.shape[1]))
	
	return X_train, X_test

def plot_pca(filepath, scaler='quantileNormal'):
	X, y, genre_mapping = loadFeatures_NoSplit(filepath)
	X = normalize(X, scaler)
	X, y = shuffle(X, y, random_state=42)
	pca = PCA().fit(X)
	fig, ax = plt.subplots(figsize=(8,6))
	x_values = range(1, pca.n_components_+1)
	ax.plot(x_values, pca.explained_variance_ratio_, lw=2, label='explained variance')
	ax.plot(x_values, np.cumsum(pca.explained_variance_ratio_), lw=2, label='cumulative explained variance')
	ax.set_title('PCA Analysis after Uniform Quantile Transformation')
	ax.set_xlabel('principal components')
	ax.set_ylabel('explained variance')
	plt.show()

def best_pca(filepath, scaler='quantileNormal'):
	X, y, genre_mapping = loadFeatures_NoSplit(filepath)
	X = normalize(X, scaler)

	logistic = LogisticRegression()
	pca = PCA()
	pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
	pca.fit(X)

	# Plot the PCA spectrum	
	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')

	# Prediction
	n_components = [27, 63, 82, 109, 145, 206] 
	#n_components = [27, 63, 82, 109, 145, 206] # Corresponding to 60-95% of variance explained (determined in test_pca)
	Cs = np.logspace(-4, 4, 3)

	# Parameters of pipelines can be set using ‘__’ separated parameter names:
	estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
	estimator.fit(X, y)

	plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
	            linestyle=':', label='n_components chosen: %s' % estimator.best_estimator_.named_steps['pca'].n_components)
	plt.legend(prop=dict(size=12))
	plt.title('Best PCA dimensionality estimated by Logistic Regression')
	plt.show()

#---------------------------------------------------LINEAR DISCRIMINANT ANALYSIS - LDA------------------------------------------------------

def test_lda(filepath):
	(X_train, y_train), (X_test, y_test), genre_mapping = load_split(filepath)
	X_train = normalize(X_train)
	lda = LDA()
	X_lda = lda.fit(X_train, y_train).transform(X_train)
	print('Number of Features after lda:', X_lda.shape[1])
	print('Explained variance lda: ', lda.explained_variance_ratio_)

def apply_lda(X_train, y_train, X_test=None):
	lda = LDA().fit(X_train, y_train)
	X_train = lda.transform(X_train)
	if X_test is not None:
		X_test = lda.transform(X_test)
		return X_train, X_test
	else:
		return X_train

def plot_lda(filepath, scaler='standard'):
    #(X_train, y_train), (X_test, y_test), genre_mapping = load_split(filepath)
    X, y, genre_mapping = loadFeatures_NoSplit(filepath)
    X = normalize(X, scaler)
    X, y = shuffle(X, y, random_state=42)

    lda_2 = LDA(n_components=2)
    X = lda_2.fit_transform(X, y)
    #X = lda_2.fit_transform(X_train, y_train)
    ax = plt.subplot(111)
    for label,marker,color in zip(range(0,8),('^', 's', 'o', '*', 'h', 'D', 'X', 'P'),('blue', 'orange', 'red', 'green', 'pink', 'purple', 'yellow', 'black')):
        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=genre_mapping[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title("Two significant vectors from LDA after Uniform Quantile Transformation")

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()	

#------------------------------------FEATURE SELECTION (Sequential Forward with kNN + Comparison PCA, NMF, kBest)------------------------------

def selector(filepath, variance=None, pca=False, scaler='quantileNormal'):
	from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
	from mlxtend.feature_selection import SequentialFeatureSelector as SFS
	
	X, y, genre_mapping = loadFeatures_NoSplit(filepath)
	X = normalize(X, scaler)

	if pca:
		pca = PCA(variance)
		X = pca.fit_transform(X)
	else:
		X = apply_lda(X, y)

	X, y = shuffle(X, y, random_state=42)

	knn = KNeighborsClassifier(n_neighbors=30)
	sfs = SFS(estimator=knn, k_features=50, forward=True, floating=False, scoring='accuracy', cv=2)
	sfs = sfs.fit(X, y)

	print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))
	print('all subsets:\n', sfs.subsets_)
	fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
	plt.title('Sequential Forward Selection std_err (quantile uniform transformation))')
	plt.grid()
	plt.show()

def test_selectors(filepath, scaler='minMax'):
	from sklearn.externals.joblib import Memory
	from sklearn.feature_selection import SelectKBest, chi2
	from tempfile import mkdtemp
	from shutil import rmtree

	features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
	X = features.iloc[:, 0:518].values
	y = features.iloc[:, 518].values

	X = normalize(X, scaler)

	enc = LabelEncoder()
	y = enc.fit_transform(y)

	X, y = shuffle(X, y, random_state=42)

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

#----------------------------------------------------------TEST CLASSIFICATION MODELS----------------------------------------------------------------

#++++++++++++++++++++Compare Classifiers: Dimensionality Reduction method (PCA vs. LDA) and Normalization method (7 scalers)++++++++++++++++++++

def save_models(df, filepath):
	if not os.path.isfile(filepath):
		df.to_csv(filepath, sep=';', decimal=',', float_format='%.3f')
	else: 
		df.to_csv(filepath, mode = 'a', header=False, sep=';', decimal=',', float_format='%.3f')

def model_testing(filepath, variance=None, pca=False, scaler='standard'):
	X, y, genre_mapping = loadFeatures_NoSplit(filepath)
	X = normalize(X, scaler)

	if pca:
		pca = PCA(variance)
		X = pca.fit_transform(X)
	else:
		X = apply_lda(X, y)

	X, y = shuffle(X, y, random_state=42)

	classsifiers = {
			'RandomForestClassifier': RandomForestClassifier(),			
			'kNN': KNeighborsClassifier(),
			'AdaBoostClassifier': AdaBoostClassifier(),
			'LR': LogisticRegression(),
			'SVM': SVC()
	}

	params = {
			'RandomForestClassifier': { 'n_estimators': [30] },
			'kNN':  { 'n_neighbors': [20] },
			'AdaBoostClassifier':  { 'n_estimators': [30] },
			'LR': [
				{'solver': ['lbfgs'], 'penalty': ['l2']}
			],
			'SVM': [
				{'kernel': ['linear'], 'C': [10]}
			]
	}

	helper = EstimatorSelectionHelper(classsifiers, params)
	f1_scorer = make_scorer(f1_score, average='micro')
	helper.fit(X, y, scoring=f1_scorer, n_jobs=2)
	summary = helper.score_summary(sort_by='min_score')
	summary['Normalizer'] = scaler

	return summary

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

#+++++++++++++++++++++Search for Best Hyperparameters for each Classification Model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def report_params(results, n_top=3):
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
              "kernel": ["linear", "rbf", "sigmoid"],
              "gamma": stats.expon(scale=.1)}
    elif model=='LR':
        param_dist = {"penalty": ["l2"],
              "C": stats.expon(scale=100),
              "solver": ["newton-cg", "sag", "lbfgs"],
              "multi_class": ["multinomial"],
              "warm_start": [True, False]}
    elif model=='kNN':
        param_dist = {"n_neighbors": stats.randint(5, 30),
              "weights": ["uniform", "distance"],
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

def tune_params(filepath, model_name):
	X, y, genre_mapping = loadFeatures(filepath)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

	X_train, X_test = normalize(X_train, X_test, scaler='quantileNormal')
	X_train = apply_lda(X_train, y_train)
	X_train, y_train = shuffle(X_train, y_train, random_state=42)

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

	random_search(X_train, y_train, model_name, model)

#+++++++++++++++++Apply Classifiers with parameters found and check quality (confusion matrix --> precision, accuracy, recall)++++++++++++++++

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Oranges):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    #
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")

    return plt

def classify_results(filepath, model, subset=None, setX=None, setY=None):
	from time import time
	import random
	
	if subset is not None:
		X, y, genre_mapping = load_subsets(filepath, subset)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
		X_train, X_test = normalize(X_train, X_test, scaler='quantileNormal')
	else:
		X, y, genre_mapping = loadFeatures(filepath)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
		X_train, X_test = normalize(X_train, X_test, scaler='quantileNormal')
		X_train, X_test = apply_lda(X_train, y_train, X_test)

	class_names = list(genre_mapping.values())

	if setX is not None:
		X_train, X_test, y_train, y_test = train_test_split(setX, setY, test_size=0.1, random_state=0)
		X_train, X_test = normalize(X_train, X_test, scaler='quantileNormal')
		subset = 'MFCC + Spectral_contrast'

	if model == 'RF':
		classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_features=1, min_samples_leaf=9, min_samples_split=3)
	elif model == 'SVM':
		classifier = SVC(kernel='sigmoid', gamma=0.008016104897209353, C=12.649805534755384)
	elif model == 'LR':
		classifier = LogisticRegression(solver='sag', penalty='l2', multi_class='multinomial', C=0.027148496216644787, warm_start=True)
	elif model == 'kNN':
		classifier = KNeighborsClassifier(algorithm='brute', leaf_size=42, n_neighbors=29, weights='uniform')
	elif model == 'MLP':
		classifier = MLPClassifier(activation='relu', alpha=0.2146079245084065, batch_size=207, solver='sgd')
	elif model == 'SGD':
		classifier = SGDClassifier(penalty=None, loss='log', alpha=0.10457337104617195, max_iter=50)
	elif model == 'extraTrees':
		classifier = ExtraTreesClassifier(n_estimators=100, criterion='entropy', max_features=2, min_samples_leaf=4, min_samples_split=4)
	else:
		raise ValueError('Classifier not recognized {}.'.format(model))

	X_train, y_train = shuffle(X_train, y_train, random_state=42)

	scoring = ['precision_micro', 'recall_micro']
	scores = cross_validate(classifier, X_train, y_train, scoring=scoring, cv=10, return_train_score=True)
	summary = pd.DataFrame.from_dict(scores)
	summary['Model'] = model

	if subset is not None:
		summary['Subset'] = subset
		titleCurve = "Learning Curves for %s using %s subset" % (model, subset)
		titleMatrix = "Confusion Matrix for %s using %s subset" % (model, subset)
	else:
		titleCurve = "Learning Curves for %s" % model
		titleMatrix = "Confusion Matrix for %s" % model
	
	start = time()
	y_pred = cross_val_predict(classifier, X_test, y_test, cv=10)
	
	if subset is not None:
		report = "Model took %.2f seconds for the %s subset, with an Accuracy of %.2f \n" % ((time() - start), subset, accuracy_score(y_test, y_pred))
	else:
		report = "Model took %.2f seconds with an Accuracy of %.2f \n" % ((time() - start), accuracy_score(y_test, y_pred))
		
	report = report + "Detailed classification report for %s : \n" % model + classification_report(y_test, y_pred) + "\n"
	print(report)

	colorMaps = ['Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'YlOrRd', 'YlOrBr', 'PuRd', 'BuGn', 'BuPu', 'YlGnBu']
	color=random.choice(colorMaps)

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, cmap=color, classes=class_names, title=titleMatrix)
	plot_learning_curve(classifier, titleCurve, X_train, y_train, ylim=(0.2, 0.95), cv=10, n_jobs=4)
	plt.show()	
	
	return report, summary

#--------------------------------------------------------------------MAIN-------------------------------------------------------------------------

if __name__ == '__main__':
	
	#Set file name and variance 
	file = 'features_final.csv'	
	
	#load_subsets(file)
	#---Plot PCA to full dataset to check variance explained
	#plot_pca(file)
	# X, y, genre_mapping = load_subsets(file, 'chroma_cens')
	# X = normalize(X, scaler='quantileUniform')
	# apply_lda(X, y)

	#---Define variance and test components selected (training set only)	
	#test_pca(file, variance=0.80)

	#---Determine the best number of components in PCA using Logistic Regression as an estimator
	#best_pca(file)

	#---Visualize LDA with just 2 components 
	#plot_lda(file, scaler='quantileUniform')

	#---Test LDA (with training set only) to check number of components chosen
	#test_lda(file)

	#---Test Sequential Forward Selection after PCA (full set, cross validation) ---> TOO LONG (63 components)
	#selector(file, variance=0.8, pca=True)

	#---Test Sequential Forward Selection after LDA (full set, cross validation)
	#selector(file, scaler='quantileUniform')

	#---Test several features selectors 
	#test_selectors(file, scaler='quantileUniform') 

	#---Test models after PCA ---> TOO LONG (63 components)
	#score = model_testing(file, variance=0.3, pca=True, scaler='quantileUniform')
	#save_models(score, 'modelSummary_pca5Comp.csv')

	#---Test models after LDA
	#model_testing(file)

	scalers = ['standard', 'minMax', 'maxAbs', 'robust', 'quantileUniform', 'quantileNormal', 'L2Norm']
	#---Test models after LDA/PCA for each method of Normalization (PCA does not converge in some cases)
	# for method in scalers:
	# 	scores = model_testing(file, scaler=method) #variance=0.95, pca=True)
	# 	save_models(scores, 'modelSummary_noReduction.csv')

	models = ['RF', 'LR', 'SVM', 'kNN', 'SGD', 'MLP', 'extraTrees']
	#---Find best hyperparameters for defined models (after LDA, with Uniform Quantile Transformation)
	#for model in models:
	#tune_params(file, model) #Save to file from console output: > file.txt

	#Train Models and get scores, confusion matrix and learning curves
	#for model in models:	
	# report, summary = classify_results(file, model)
	# with open("Split_Scores.txt", "a") as file:
	# 	file.write(report)
	# save_models(summary, 'Split_Scores.csv')

	#Train Models and get scores, confusion matrix and learning curves for SUBSETS or COMBO
	subsets = ['chroma_cens', 'chroma_stft', 'chroma_cqt', 'spec_rolloff', 'spec_contrast', 'spec_centroid', 'spec_bandwidth', 'tonnetz', 'rmse', 'mfcc', 'zcr']
	combo = ['mfcc', 'spec_contrast']
	#X, y = combine_subsets(file, combo)
	#for subset in subsets:
	report, summary = classify_results(file, 'SGD') #, subset=subset) #, setX=X, setY=y)
		# with open("Results_Subsets.txt", "a") as file:
		# 	file.write(report)
		# save_models(summary, 'Results_Subsets.csv')



