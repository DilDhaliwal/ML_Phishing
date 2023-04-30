from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
class Tree:

    def __init__(self, X_train, y_train, X_test, y_test, max_depth, min_samples_split, min_samples_leaf):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeClassifier(random_state = RANDOM_STATE, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)

    def fit_tree(self):
        self.tree.fit(self.X_train, self.y_train)
        return
    
    def predict_tree(self):
        self.y_pred = self.tree.predict(self.X_test)
        return self.y_pred
    
    def accuracy_tree(self):
        self.score = self.tree.score(self.X_test, self.y_test)
        return self.score
    
    def metrics_tree(self):
        y_pred = self.predict_tree()
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        f1 = f1_score(self.y_test, y_pred, average='macro')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        return accuracy, precision, recall, f1
    

        