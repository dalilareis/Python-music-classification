
import os
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.model_selection import GridSearchCV, cross_val_predict, learning_curve, train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


#------------------------------------------------------LOAD FILES, PRE-PROCESS AND NORMALIZE-------------------------------------------------------------

def load_split(filepath='features_final.csv'):
	#Load csv file
	features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2]) #index_col=0 uses 1st column as row names (starts counting at 2nd column)

	#Get labels from data and split data into train and test sets
	X = features.iloc[:, 0:518].values
	y = features.iloc[:, 518].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
	X_train, y_train = shuffle(X_train, y_train, random_state=42)

	#Transform genres to integer labels
	enc = LabelEncoder()
	y_train = enc.fit_transform(y_train)
	y_test = enc.transform(y_test)
	genre_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_, ))

	return (X_train, y_train), (X_test, y_test), genre_mapping

def loadFeatures_NoSplit(filepath='features_final.csv'):
	features = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
	X = features.iloc[:, 0:518].values
	y = features.iloc[:, 518].values

	enc = LabelEncoder()
	y = enc.fit_transform(y)
	genre_mapping = dict(zip(enc.transform(enc.classes_), enc.classes_, ))
	X, y = shuffle(X, y, random_state=42)

	return X, y, genre_mapping

def load_subsets(subset, filepath='features_final.csv'):
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

def combine_subsets(listSubsets):
	listX = []

	for subset in listSubsets:
		X, y, genre_mapping = load_subsets(subset)
		listX.append(X)

	X = np.concatenate((listX), axis=1)

	return X

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
		
#---------------------------------------------------PRINCIPAL COMPONENT ANALYSIS - PCA-------------------------------------------------------

def test_pca(scaler='quantileNormal', variance=0.75):
    (X_train, y_train), (X_test, y_test), genre_mapping = load_split()
    X_train = normalize(X_train, scaler=scaler)
    pca = PCA(variance)
    pca.fit_transform(X_train)
    selected = pca.n_components_
    report = "Selected %s principal components from %s original features \n for %.2f variance, using %s scaler" % (selected, X_train.shape[1], variance, scaler)
    return report
    #95% variance --> 206 components 	90% --> 145		85% --> 109 	80% --> 82 		75% --> 63		60% --> 27
 
def apply_pca(train, test, variance):
	pca = PCA(variance)
	X_train = pca.fit_transform(train)
	X_test = pca.transform(test)
	
	return X_train, X_test

def plot_pca():
	X, y, genre_mapping = loadFeatures_NoSplit()
	X = normalize(X)
	X, y = shuffle(X, y, random_state=42)
	pca = PCA().fit(X)
	fig, ax = plt.subplots(figsize=(8,6))
	x_values = range(1, pca.n_components_+1)
	ax.plot(x_values, pca.explained_variance_ratio_, lw=2, label='explained variance')
	ax.plot(x_values, np.cumsum(pca.explained_variance_ratio_), lw=2, label='cumulative explained variance')
	ax.set_title('PCA Analysis')
	ax.set_xlabel('principal components')
	ax.set_ylabel('explained variance')
	plt.show()

#---------------------------------------------------LINEAR DISCRIMINANT ANALYSIS - LDA------------------------------------------------------

def test_lda(scaler='quantileNormal'):
	(X_train, y_train), (X_test, y_test), genre_mapping = load_split()
	X_train = normalize(X_train, scaler=scaler)
	lda = LDA()
	X_lda = lda.fit(X_train, y_train).transform(X_train)
	report = "Features after LDA using %s scaler = %s" % (scaler, X_lda.shape[1])
	report = report + "\nExplained variance lda: %s" %lda.explained_variance_ratio_
	return report

def apply_lda(X_train, y_train):
	lda = LDA()
	X_lda = lda.fit(X_train, y_train).transform(X_train)

	return X_lda

def plot_lda():
    #(X_train, y_train), (X_test, y_test), genre_mapping = load_split(filepath)
    X, y, genre_mapping = loadFeatures_NoSplit()
    X = normalize(X)
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
    plt.title("Two significant vectors from LDA after quantile Uniform transformation")

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

#----------------------------------------------------------TEST CLASSIFICATION MODELS----------------------------------------------------------------

#++++++++++++++++++++Compare Classifiers: Dimensionality Reduction method (PCA vs. LDA) and Normalization method (7 scalers)++++++++++++++++++++

def save_models(df, filepath):
	if not os.path.isfile(filepath):
		df.to_csv(filepath, sep=';', decimal=',', float_format='%.3f')
	else: 
		df.to_csv(filepath, mode = 'a', header=False, sep=';', decimal=',', float_format='%.3f')

def model_testing(variance=None, pca=False, scaler='standard'):
	X, y, genre_mapping = loadFeatures_NoSplit()
	X = normalize(X, scaler)

	if pca:
		pca = PCA(variance)
		X = pca.fit_transform(X)
		file = "modelSummary_PCA_%s.csv" % variance
	else:
		X = apply_lda(X, y)
		file = "modelSummary_LDA.csv"

	X, y = shuffle(X, y, random_state=42)

	classsifiers = {
			'RandomForestClassifier': RandomForestClassifier(),			
			'kNN': KNeighborsClassifier(),
			'AdaBoostClassifier': AdaBoostClassifier(),
			'LR': LogisticRegression(),
			'SVM': SVC()
	}

	params = {
			'RandomForestClassifier': { 'n_estimators': [30, 100] },
			'kNN':  { 'n_neighbors': [8, 20] },
			'AdaBoostClassifier':  { 'n_estimators': [30, 100] },
			'LR': [
				{'solver': ['saga'], 'penalty': ['l1']},
				{'solver': ['lbfgs'], 'penalty': ['l2']}
			],
			'SVM': [
				{'kernel': ['linear'], 'C': [1, 10]},
				{'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}
			]
	}

	helper = EstimatorSelectionHelper(classsifiers, params)
	f1_scorer = make_scorer(f1_score, average='micro')
	helper.fit(X, y, scoring=f1_scorer, n_jobs=2)
	summary = helper.score_summary(sort_by='min_score')
	summary['Normalizer'] = scaler

	save_models(summary, file)

	return file

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

#+++++++++++++++++Apply Classifiers with parameters found and check quality (confusion matrix --> precision, accuracy, recall)++++++++++++++++

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.YlGnBu):

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

def classify_results(model, scaler='quantileNormal', subset=None, setX=None):
	from time import time
	import random
	
	if subset is not None:
		X, y, genre_mapping = load_subsets(subset)
		X = normalize(X, scaler=scaler)
	else:
		X, y, genre_mapping = loadFeatures_NoSplit()
		X = normalize(X, scaler=scaler)
		X = apply_lda(X, y)

	class_names = list(genre_mapping.values())

	if setX is not None:
		X = normalize(setX, scaler=scaler)
		#X = apply_lda(X, y)
		subset = 'MFCC + Spec_Contrast'

	if model == 'RF':
		classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features=1, min_samples_leaf=6, min_samples_split=5)
	elif model == 'SVM':
		classifier = SVC(kernel='linear', gamma=0.15043359874882642, C=37.04299991666622)
	elif model == 'LR':
		classifier = LogisticRegression(solver='lbfgs', penalty='l2', C=12.860378815138466, multi_class='multinomial', warm_start=True)
	elif model == 'kNN':
		classifier = KNeighborsClassifier(algorithm='ball_tree', leaf_size=31, n_neighbors=29)
	elif model == 'MLP':
		classifier = MLPClassifier(activation='relu', alpha=0.05440165708835292, batch_size=358, solver='adam')
	elif model == 'SGD':
		classifier = SGDClassifier(penalty='elasticnet', loss='log', alpha=0.02011798244191186, max_iter=50)
	elif model == 'extraTrees':
		classifier = ExtraTreesClassifier(n_estimators=50, criterion='gini', max_features=3, min_samples_leaf=5, min_samples_split=8)
	else:
		raise ValueError('Classifier not recognized {}.'.format(model))

	X, y = shuffle(X, y, random_state=42)

	if subset is not None:
		titleCurve = "Learning Curves for %s using %s subset" % (model, subset)
		titleMatrix = "Confusion Matrix for %s using %s subset" % (model, subset)
	else:
		titleCurve = "Learning Curves for %s" % model
		titleMatrix = "Confusion Matrix for %s" % model
	
	start = time()
	y_pred = cross_val_predict(classifier, X, y, cv=10)
	if subset is not None:
		report = "Model took %.2f seconds for the %s subset, with an Accuracy of %.4f \n" % ((time() - start), subset, accuracy_score(y, y_pred))
	else:
		report = "Model took %.2f seconds with an Accuracy of %.4f \n" % ((time() - start), accuracy_score(y, y_pred))
		
	report = report + "Detailed classification report for %s : \n" % model + classification_report(y, y_pred) + "\n"

	colorMaps = ['Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'YlOrRd', 'YlOrBr', 'PuRd', 'BuGn', 'BuPu', 'YlGnBu']
	color=random.choice(colorMaps)

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y, y_pred)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, title=titleMatrix, cmap=color)
	plot_learning_curve(classifier, titleCurve, X, y, ylim=(0.2, 1.01), cv=10, n_jobs=2)
	plt.show()

	return report

def fit_model(model, scaler='quantileNormal'):
	#Usar todos os dados para garantir a representacao de todos os genres
	X, y, genre_mapping = loadFeatures_NoSplit()

	if scaler=='standard':
		norm = StandardScaler()
	elif scaler=='minMax':
		norm = MinMaxScaler()
	elif scaler=='maxAbs':
		norm = MaxAbsScaler()
	elif scaler=='robust':
		norm = RobustScaler(quantile_range=(25, 75))
	elif scaler=='quantileUniform':
		norm = QuantileTransformer(output_distribution='uniform')
	elif scaler=='quantileNormal':
		norm = QuantileTransformer(output_distribution='normal')
	elif scaler=='L2Norm':
		norm = Normalizer()

	if model == 'RF':
		classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features=1, min_samples_leaf=6, min_samples_split=5)
	elif model == 'SVM':
		classifier = SVC(kernel='linear', gamma=0.15043359874882642, C=37.04299991666622)
	elif model == 'LR':
		classifier = LogisticRegression(solver='lbfgs', penalty='l2', C=12.860378815138466, multi_class='multinomial', warm_start=True)
	elif model == 'kNN':
		classifier = KNeighborsClassifier(algorithm='ball_tree', leaf_size=31, n_neighbors=29)
	elif model == 'MLP':
		classifier = MLPClassifier(activation='relu', alpha=0.05440165708835292, batch_size=358, solver='adam')
	elif model == 'SGD':
		classifier = SGDClassifier(penalty='elasticnet', loss='log', alpha=0.02011798244191186, max_iter=50)
	elif model == 'extraTrees':
		classifier = ExtraTreesClassifier(n_estimators=50, criterion='gini', max_features=3, min_samples_leaf=5, min_samples_split=8)
	else:
		raise ValueError('Classifier not recognized {}.'.format(model))

	#Usar e guardar o pipeline para aplicar os mesmos passos aos novos dados de teste
	pipe = Pipeline([('normalize', norm), ('reduce_dim', LDA()),('classify', classifier)])

	pipe.fit(X, y)

	#Previsoes com dados novos
	# score = pipe.score(X,y)
	# features = pd.read_csv('sampleFeatures.csv', index_col=0, header=[0, 1, 2])
	# X = features.iloc[:, 0:518].values
	# predict = pipe.predict(X)
	# print("Score is %s and Prediction is %s with model %s" % (score, predict, model))

	joblib.dump(pipe, 'final_model.pkl')
	result = "%s Model using %s scaler" % (model, scaler)
	return result

#--------------------------------------------------------------------MAIN-------------------------------------------------------------------------

if __name__ == "__main__":

	fit_model('extraTrees')

