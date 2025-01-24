import math
from collections import Counter
from sklearn.datasets import load_iris
import random

def builtin_function(iris):
    x, y = iris.data, iris.target

    x_builtin_train, x_builtin_test, y_builtin_train, y_builtin_test = train_test_split(x, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)

    clf.fit(x_builtin_train, y_builtin_train)

    y_builtin_pred = clf.predict(x_builtin_test)

    accuracy = accuracy_score(y_builtin_test, y_builtin_pred)
    
    return accuracy

    # print(f"Accuracy of the sklearn library: {accuracy:.2f}")

    # plt.figure(figsize=(12, 8))
    # plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    # plt.title("Decision Tree")
    # plt.show()

# Load Iris dataset
def prepare_dataset(iris):
    data = []
    for features, label in zip(iris.data, iris.target):
        data.append(list(features) + [label])
    return data

# Split dataset into training and test data
def train_test_split(data, test_ratio=0.2):
    train_size = int(len(data) * (1 - test_ratio))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Calculate Gini Impurity
def gini_impurity(groups, classes):
    total_samples = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        score = 0.0
        group_labels = [row[-1] for row in group]
        for class_val in classes:
            proportion = group_labels.count(class_val) / len(group)
            score += proportion ** 2
        gini += (1 - score) * (len(group) / total_samples)
    return gini

# Split data based on a feature and its threshold
def split_data(index, threshold, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Find the best split for a dataset
def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_threshold, best_score, best_groups = None, None, float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_data(index, row[index], dataset)
            gini = gini_impurity(groups, class_values)
            if gini < best_score:
                best_index, best_threshold, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'threshold': best_threshold, 'groups': best_groups}

# Create a terminal node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return Counter(outcomes).most_common(1)[0][0]

# Build the decision tree
class DecisionTree:
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size
        self.tree = None

    def fit(self, train_data):
        self.tree = self._build_tree(get_best_split(train_data), 1)

    def _build_tree(self, node, depth):
        left, right = node['groups']
        del node['groups']

        # Check for a no split
        if not left or not right:
            node['left'] = node['right'] = to_terminal(left + right)
            return node

        # Check maximum depth
        if depth >= self.max_depth:
            node['left'], node['right'] = to_terminal(left), to_terminal(right)
            return node

        # Process left child
        if len(left) <= self.min_size:
            node['left'] = to_terminal(left)
        else:
            node['left'] = self._build_tree(get_best_split(left), depth + 1)

        # Process right child
        if len(right) <= self.min_size:
            node['right'] = to_terminal(right)
        else:
            node['right'] = self._build_tree(get_best_split(right), depth + 1)

        return node

    def predict(self, row):
        return self._predict_row(self.tree, row)

    def _predict_row(self, node, row):
        if row[node['index']] < node['threshold']:
            if isinstance(node['left'], dict):
                return self._predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict_row(node['right'], row)
            else:
                return node['right']

# Evaluate the Decision Tree
def evaluate_model(tree, test_data):
    predictions = [tree.predict(row) for row in test_data]
    actual = [row[-1] for row in test_data]
    accuracy = sum(1 for p, a in zip(predictions, actual) if p == a) / len(actual)
    return accuracy

# Main execution
if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    data = prepare_dataset(iris)

    # Shuffle and split the dataset
    random.shuffle(data)
    train_data, test_data = train_test_split(data)

    # Train the Decision Tree
    max_depth = 3
    min_size = 6
    tree = DecisionTree(max_depth, min_size)
    tree.fit(train_data)

    # Evaluate the model
    accuracy = evaluate_model(tree, test_data)
    print(f"Accuracy: {accuracy * 100:.2f}%")
