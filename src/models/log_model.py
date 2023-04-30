from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

class Log:

    def __init__(self, X_train, y_train, X_test, y_test, c):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.c = c
        self.log = LogisticRegression(C = c, max_iter=15000, random_state=RANDOM_STATE)

    def fit_log(self):
        self.log.fit(self.X_train, self.y_train)
        return 
    
    def predict_log(self):
        self.y_pred = self.log.predict(self.X_test)
        return self.y_pred
    
    def accuracy_log(self):
        self.score = self.log.score(self.X_test, self.y_test)
        return self.score
    
    def metrics_log(self):
        y_pred = self.predict_log()
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        f1 = f1_score(self.y_test, y_pred, average='macro')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        return accuracy, precision, recall, f1
    