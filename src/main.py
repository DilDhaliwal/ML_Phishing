#IMPORTS

from sklearn.metrics import roc_curve

from data.pre_processing import PreProcess
from models.log_model import Log
from data.pre_processing import LoadData
from visualization.visualize import Visualize
from models.tree_model import Tree
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, random
from sklearn import tree
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#RANDOM STATE
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

#IMPORT DATASET
print('------------------------------------------------------------')
URL = 'https://raw.githubusercontent.com/DilDhaliwal/CSV-for-322/main/Phishing_Legitimate_full.csv'
print('DATASET LOADING')
LD = LoadData(URL)
print('DATASET LOADED')
df = LD.readcsv()
print('------------------------------------------------------------')
print(df.dtypes)

#DATA PROCESSING
print('------------------------------------------------------------')
X = df.drop(['id', 'CLASS_LABEL'], axis = 1)
y = df['CLASS_LABEL'].array
print('DATASET SPLIT INTO X/y')
print('DATASET PREPROCESSING')
pre = PreProcess(df, X)
pre.check_null()
X = pre.scale()
print('DATASET PREPROCESSED (FEATURES SCALED AND DENULLED)') 
print('DATASET SPLITTING INTO TRAIN/TEST') 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
print('------------------------------------------------------------')

#LOGISTIC REGRESSION MODEL
print('------------------------------------------------------------')
print('LOGISTIC REGRESSION MODELLED (L2 REGULARIZED) \n')
log = Log(X_train, y_train, X_test, y_test, c = 0.000012)
log.fit_log()
y_pred = log.predict_log()
cf_matrix_l = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (LR): \n',cf_matrix_l)
print('\nLOGISTIC REGRESSION METRICS')
print(log.metrics_log())
print('------------------------------------------------------------')

#DECISION TREE MODEL
print('------------------------------------------------------------')
print('DECISION TREE MODELLED (PRE-PRUNED) \n')
tree = Tree(X_train, y_train, X_test, y_test, max_depth = 3, min_samples_split = 2, min_samples_leaf = 2)
tree.fit_tree()
# 20, 0.3, 0.3
y_pred = tree.predict_tree()
cf_matrix_t = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (DT): \n', cf_matrix_t)
print('\nDECISION TREE METRICS')
print(tree.metrics_tree())
print('------------------------------------------------------------')

#KFOLD VALIDATION
print('------------------------------------------------------------')
print('LOGISTIC REGRESSION MODEL KFOLD VALIDATION')
cv = RepeatedKFold(n_splits=8, n_repeats=4, random_state=RANDOM_STATE)
scores = cross_val_score(log.log, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
print('------------------------------------------------------------')
print('------------------------------------------------------------')
print('DECISION TREE MODEL KFOLD VALIDATION')
cv = RepeatedKFold(n_splits=8, n_repeats=4, random_state=RANDOM_STATE)
scores = cross_val_score(tree.tree, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
print('------------------------------------------------------------')

#VISUALIZATIONS
print('------------------------------------------------------------')
print('VISUALIZATIONS')
y_score_t = tree.tree.predict_proba(X_test)[:,1]
y_score_l = log.log.predict_proba(X_test)[:,1]
false_positive_rate_t, true_positive_rate_t, threshold_t = roc_curve(y_test, y_score_t)
false_positive_rate_l, true_positive_rate_l, threshold_l = roc_curve(y_test, y_score_l)

vis = Visualize(df, X_train, y_train, X_test, y_test, tree.tree, cf_matrix_l, cf_matrix_t, false_positive_rate_l, true_positive_rate_l, false_positive_rate_t, true_positive_rate_t)
vis.corr_matrix()
vis.tree_plot()
vis.plot_importance()
vis.tree_matrix()
vis.roc_tree()
vis.log_matrix()
vis.roc_log()