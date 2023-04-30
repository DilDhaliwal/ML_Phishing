import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import random
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

class Visualize:

    def __init__(self, data, X_train, y_train, X_test, y_test, tree, log_cfm, tree_cfm, fp_l, tp_l, fp_t, tp_t):
        self.data = data
        self.tree = tree
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.log_cfm = log_cfm
        self.tree_cfm = tree_cfm
        self.fp_l = fp_l
        self.tp_l = tp_l
        self.fp_t = fp_t
        self.tp_t = tp_t

    def tree_plot(self):
        plt.figure(figsize=(12,12))
        tree.plot_tree(self.tree, filled=True)
        plt.title('DecisionTree')
        plt.show()

    def plot_importance(self):
        Importance = pd.DataFrame({'Importance':self.tree.feature_importances_*100}, index=self.X_train.columns)
        Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
        plt.xlabel('Variable Importance')
        plt.gca().legend_ = None
        plt.title('Feature Importance - DecisionTree')
        plt.show()
    
    def log_matrix(self):
        cm_display = metrics.ConfusionMatrixDisplay(self.log_cfm, display_labels = [False, True])
        cm_display.plot()
        plt.title('Confusion Matrix - LogisticRegression')
        plt.show()

    def tree_matrix(self):
        cm_display = metrics.ConfusionMatrixDisplay(self.tree_cfm, display_labels = [False, True])
        cm_display.plot()
        plt.title('Confusion Matrix - DecisionTree')
        plt.show()

    def roc_log(self):
        plt.subplots(1, figsize=(10,10))
        plt.title('Receiver Operating Characteristic - LogisticRegression')
        plt.plot(self.fp_l, self.tp_l)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def roc_tree(self):
        plt.subplots(1, figsize=(10,10))
        plt.title('Receiver Operating Characteristic - DecisionTree')
        plt.plot(self.fp_t, self.tp_t)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def corr_matrix(self):
        corrmax = self.data.corr()
        f, ax = plt.subplots(figsize=(12,15))
        sns.heatmap(corrmax)
        plt.show()