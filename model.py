class NaiveBayes:
    def __init__(self):
        self.class_prob = {}
        self.feature_prob = {}

    def fit(self, X_train, y_train):
        # Menghitung probabilitas kelas dan fitur
        total_samples = len(y_train)
        self.class_prob = {label: sum(y_train == label) / total_samples for label in set(y_train)}

        n_features = X_train.shape[1]
        self.feature_prob = {}
        for label in set(y_train):
            label_indices = y_train == label
            label_features = X_train[label_indices]
            self.feature_prob[label] = {}
            for feature in range(n_features):
                feature_values = label_features.iloc[:, feature]
                self.feature_prob[label][feature] = {
                    value: (sum(feature_values == value) + 1) / (len(label_features) + n_features)
                    for value in set(feature_values)
                }

    def predict(self, X_test):
        predictions = []
        for sample in X_test.values:
            max_class = None
            max_prob = -1
            for label, class_prob in self.class_prob.items():
                prob = class_prob
                for idx, value in enumerate(sample):
                    if value in self.feature_prob[label][idx]:
                        prob *= self.feature_prob[label][idx][value]
                    else:
                        prob *= 1 / (len(self.feature_prob[label][idx]) + 1)
                if prob > max_prob:
                    max_prob = prob
                    max_class = label
            predictions.append(max_class)
        return predictions

    
class DecisionTree:
    def __init__(self):
        self.tree = None

    def _calculate_gini_index(self, groups, classes):
        n_instances = sum(len(group) for group in groups)
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def _split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def _get_best_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self._split(index, row[index], dataset)
                gini = self._calculate_gini_index(groups, class_values)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
        return {'index':best_index, 'value':best_value, 'groups':best_groups}

    def _to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def _split_node(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self._to_terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_best_split(left)
            self._split_node(node['left'], max_depth, min_size, depth+1)
        if len(right) <= min_size:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_best_split(right)
            self._split_node(node['right'], max_depth, min_size, depth+1)

    def _build_tree(self, train, max_depth=3, min_size=1):
        root = self._get_best_split(train)
        self._split_node(root, max_depth, min_size, 1)
        return root

    def _predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

    def fit(self, X_train, y_train):
        dataset = X_train.copy()
        dataset['class'] = y_train
        train = dataset.values.tolist()
        self.tree = self._build_tree(train)

    def predict(self, X_test):
        predictions = []
        for row in X_test.values.tolist():
            prediction = self._predict(self.tree, row)
            predictions.append(prediction)
        return predictions

 

import random

class RandomForest:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X_train, y_train):
        for _ in range(self.n_trees):
            # Membuat pohon keputusan acak
            decision_tree = DecisionTree()
            decision_tree.fit(X_train, y_train)
            self.trees.append(decision_tree)

    def predict(self, X_test):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X_test))
        # mengembalikan mode dari prediksi semua pohon
        return [max(set(column), key=column.count) for column in zip(*predictions)]

   

import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class EvaluationMetrics:
    def __init__(self):
        self.accuracies = []
        self.confusion_matrices = []
        self.execution_time = None
        self.average_accuracy = None
        self.overall_confusion_matrix = None

    def calculate_overall_metrics(self):
        # Calculate average accuracy
        self.average_accuracy = np.mean(self.accuracies)

        # Calculate overall confusion matrix
        self.overall_confusion_matrix = sum(self.confusion_matrices)
    def evaluate_model(self, model, X, y, split_ratio, num_splits):
        # Placeholder for storing evaluation results
        accuracies = []
        confusion_matrices = []
        execution_times = []
        

        for fold in range(num_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end_time = time.time()

            # Calculate accuracy for each split
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            # Calculate confusion matrix for each split
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices.append(cm)

            # Calculate execution time for each split
            execution_times.append(end_time - start_time)

            # Save accuracy and confusion matrix for each fold
            self.accuracies.append(accuracy)
            self.confusion_matrices.append(cm)


        # Aggregate evaluation results
        self.execution_time = np.mean(execution_times)

        # Calculate overall metrics
        self.calculate_overall_metrics()
        return self.accuracies, self.confusion_matrices, self.execution_time



 



