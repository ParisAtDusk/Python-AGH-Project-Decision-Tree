import unittest
from sklearn.datasets import load_iris
from collections import Counter
from random import shuffle
from statistics import mean
from iris import DecisionTree, my_train_test_split, gini_impurity, to_terminal_node, test_split, get_best_split, accuracy_metric

class TestDecisionTree(unittest.TestCase):

    def setup(self):
        """Set up the dataset for testing."""
        iris = load_iris()
        self.data = [list(row) + [label] for row, label in zip(iris.data, iris.target)]
        self.feature_names = iris.feature_names
        self.class_names = iris.target_names

    def test_gini_impurity(self):
        """Test the Gini impurity calculation."""
        # [data point, class]
        groups = [[[1, 0], [2, 0]], [[3, 1], [4, 1]]]   # pure groups
        classes = [0, 1]
        gini = gini_impurity(groups, classes)
        self.assertAlmostEqual(gini, 0.0, places=4)

        groups = [[[1, 0], [2, 1]], [[3, 0], [4, 1]]]   # mixed groups
        gini = gini_impurity(groups, classes)
        self.assertGreater(gini, 0.0)

    def test_to_terminal(self):
        """Test the terminal node determination."""
        group = [[1, 0], [2, 0], [3, 1]]
        terminal = to_terminal_node(group)
        self.assertEqual(terminal, 0)  # Most common class is 0

    def test_train_test_split(self):
        """Test custom train-test split function."""
        shuffle(self.data)
        train_data, test_data = my_train_test_split(self.data, test_ratio=0.2)
        self.assertEqual(len(train_data), int(len(self.data) * 0.8))
        self.assertEqual(len(test_data), int(len(self.data) * 0.2))

    def test_best_split(self):
        """Test finding the best split."""
        split = get_best_split(self.data)
        self.assertIn('index', split)
        self.assertIn('threshold', split)
        self.assertIn('groups', split)
        self.assertIn('gini', split)

    def test_decision_tree_training(self):
        """Test training the decision tree."""
        shuffle(self.data)
        train_data, _ = my_train_test_split(self.data, test_ratio=0.2)
        tree = DecisionTree(max_depth=10, min_size=1)
        tree.fit(train_data)
        self.assertIsNotNone(tree.tree)
        self.assertIn('index', tree.tree)

    def test_decision_tree_prediction(self):
        """Test prediction with the decision tree."""
        shuffle(self.data)
        train_data, test_data = my_train_test_split(self.data, test_ratio=0.2)
        tree = DecisionTree(max_depth=10, min_size=1)
        tree.fit(train_data)
        predictions = [tree.predict(row[:-1]) for row in test_data]
        self.assertEqual(len(predictions), len(test_data))

    def test_accuracy_metric(self):
        """Test accuracy calculation."""
        actual = [0, 1, 1, 0, 2]
        predicted = [0, 1, 1, 0, 2]
        accuracy = accuracy_metric(actual, predicted)
        self.assertEqual(accuracy, 100.0)

        predicted = [0, 1, 0, 0, 2]
        accuracy = accuracy_metric(actual, predicted)
        self.assertLess(accuracy, 100.0)

    def test_full_pipeline(self):
        """Test the entire training and evaluation pipeline."""
        accuracies = []
        for _ in range(5):
            shuffle(self.data)
            train_data, test_data = my_train_test_split(self.data, test_ratio=0.2)
            train_x, train_y = [row[:-1] for row in train_data], [row[-1] for row in train_data]
            test_x, test_y = [row[:-1] for row in test_data], [row[-1] for row in test_data]

            tree = DecisionTree(max_depth=10, min_size=1)
            tree.fit(train_data)

            predictions = [tree.predict(row) for row in test_data]
            accuracy = accuracy_metric(test_y, predictions)
            accuracies.append(accuracy)

        avg_accuracy = mean(accuracies)
        self.assertGreater(avg_accuracy, 80.0)  # Expect reasonable accuracy

if __name__ == "__main__":
    unittest.main()
