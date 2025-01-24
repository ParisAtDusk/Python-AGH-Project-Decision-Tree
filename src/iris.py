from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from random import shuffle, randrange, seed
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from statistics import mean 

def builtin_function(m_depth, rd_state, train_x, train_y, test_x, test_y):

    # x_builtin_train, x_builtin_test, y_builtin_train, y_builtin_test = train_test_split(x, y, test_size=0.2)

    sk_tree = DecisionTreeClassifier(max_depth=m_depth,random_state=rd_state)
    sk_tree.fit(train_x, train_y)
    sk_pred = sk_tree.predict(test_x)

    accuracy = accuracy_score(test_y, sk_pred)


    print(f"Accuracy sklearn: {accuracy*100:.2f}%")

    plt.figure(figsize=(12, 8))
    plot_tree(sk_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.title("Decision Tree")
    plt.show()


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

def to_terminal_node(group):
    outcomes = [row[-1] for row in group]
    return Counter(outcomes).most_common(1)[0][0]       # return most common group

def my_train_test_split(data, test_ratio=0.2):
    train_size = int(len(data) * (1 - test_ratio))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

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

        # Compute and store class distribution and sample count
        classes = list(set(row[-1] for row in left + right))
        node['samples'] = len(left) + len(right)
        # node['value'] = [sum(1 for row in left + right if row[-1] == c) for c in range(len(set(row[-1] for row in left + right)))]
        # node['value'] = [sum(1 for row in left + right if row[-1] == c) for c in classes]
        node['gini'] = gini_impurity([left, right], list(set(row[-1] for row in left + right)))
        node['value'] = [0] * 3
        for row in left + right:
          node['value'][row[-1]] += 1  # Increment count for the corresponding class
  
         # Stop splitting if Gini impurity is 0
        if node['gini'] == 0:
            node['left'] = node['right'] = to_terminal_node(left + right)
            return node

        # No split
        # if not left or not right:
        #     node['left'] = node['right'] = to_terminal(left + right)
        #     return node

        # Max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = to_terminal_node(left), to_terminal_node(right)
            return node

        # Left node
        if len(left) <= self.min_size:
            node['left'] = to_terminal_node(left)
        else:
            node['left'] = self._build_tree(get_best_split(left), depth + 1)

        # Right node
        if len(right) <= self.min_size:
            node['right'] = to_terminal_node(right)
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

# Split a dataset left and right based on threshold value
def test_split(index, threshold, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Select the best split point for a dataset
def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_threshold, best_score, best_groups = None, None, float('inf'), None

    for index in range(len(dataset[0]) - 1):
        sorted_values = sorted(set(row[index] for row in dataset))
        thresholds = [(sorted_values[i] + sorted_values[i+1]) / 2 for i in range(len(sorted_values) - 1)]
        for threshold in thresholds:
            groups = test_split(index, threshold, dataset)
            gini = gini_impurity(groups, class_values)
            if gini < best_score:
                best_index, best_threshold, best_score, best_groups = index, threshold, gini, groups

    # Add Gini impurity and sample size to the returned node
    return {
        'index': best_index,
        'threshold': best_threshold,
        'groups': best_groups,
        'gini': best_score,
        'samples': len(dataset)
    }


# Predict a dataset recursively 
def predict(tree, row):
    if row[tree['index']] < tree['threshold']:      # Check if the row feature is less than threshold 
        if isinstance(tree['left'], dict):          # Then check if the feature has a child node
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']

# Calculate accuracy
def accuracy_metric(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual) * 100.0        # list of predicted labels  / list of actual class labels


def visualize_tree(tree, feature_names, class_names):
    def calculate_positions(node, depth=0, x_pos=0):
        """
        Recursively calculate x/y positions for each node.
        """
        if isinstance(node, dict):
            # For decision nodes
            left_width = calculate_positions(node['left'], depth + 1, x_pos)
            right_width = calculate_positions(node['right'], depth + 1, x_pos + left_width)
            node_x = x_pos + left_width / 2
            positions[id(node)] = (node_x, -depth)
            return left_width + right_width
        else:
            # For leaf nodes
            positions[id(node)] = (x_pos, -depth)
            return 1  # One unit for leaf nodes

    def draw_node(ax, node_id, x, y, label, color):
        """
        Draw a single node (box or ellipse).
        """
        box = FancyBboxPatch((x - 0.3, y - 0.05), 0.6, 0.1,boxstyle="square,pad=0.2", fc=color, ec="black")
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=6)

    def draw_edge(ax, x1, y1, x2, y2, label):
        """
        Draw an edge between two nodes with optional label.
        """
        ax.plot([x1, x2], [y1-0.25, y2+0.25], 'k-', linewidth=1)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=6, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white"))

    def build_tree_plot(node, parent_coords=None, edge_label=""):
        """
        Recursively plot nodes and edges.
        """
        color=["orange", "lightgreen", "blue"]
        node_coords = positions[id(node)]
        x, y = node_coords

        if parent_coords is not None:
            parent_x, parent_y = parent_coords
            draw_edge(ax, parent_x, parent_y, x, y, edge_label)

        if isinstance(node, dict) and node['value'] != None:
            if node["left"] == node["right"]:
                label = (f"class = {class_names[node['value'].index(max(node['value']))]}\n"
                        f"gini = {node['gini']:.4f}\n"
                        f"samples = {node['samples']}\n"
                        f"value = {node['value']}")
            else:
                label = (f"{feature_names[node['index']]} <= {node['threshold']:.2f}\n"
                        f"class = {class_names[node['value'].index(max(node['value']))]}\n"
                        f"gini = {node['gini']:.4f}\n"
                        f"samples = {node['samples']}\n"
                        f"value = {node['value']}")
            draw_node(ax, id(node), x, y, label, color=color[node['value'].index(max(node['value']))])

            if node["left"] != node["right"]:
                build_tree_plot(node["left"], node_coords, "True")
                build_tree_plot(node["right"], node_coords, "False")
        else:
            label = (f"class = {class_names[node]}\n")
                        # f"gini = {node['gini']:.4f}\n"
                        # f"samples = {node['samples']}\n"
                        # f"value = {node['value']}")
            draw_node(ax, id(node), x, y, label, color=color[node])
    # Step 1: Calculate positions for all nodes
    positions = {}
    calculate_positions(tree.tree)

    # Step 2: Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]
    ax.set_xlim(min(x_coords) - 0.5, max(x_coords) + 0.5)
    ax.set_ylim(min(y_coords) - 0.5, max(y_coords) + 0.5)
    ax.axis("off")

    # Step 3: Build the plot
    build_tree_plot(tree.tree)
    plt.ion()
    plt.show()


if __name__ == "__main__":
    iris = load_iris()
    data = [list(row) + [label] for row, label in zip(iris.data, iris.target)]
    max_depth = 20
    # seed(10)
    # seed(6)
    # shuffle(data)

    # Split the data for testing and training. 20% goes for testing
    train_data, test_data = my_train_test_split(data, test_ratio=0.2)
    train_x, train_y = [row[:-1] for row in train_data], [row[-1] for row in train_data]
    test_x, test_y = [row[:-1] for row in test_data], [row[-1] for row in test_data]

    # Train the decision tree
    tree = DecisionTree(max_depth=max_depth, min_size=1)    
    tree.fit(train_data)

    # Make predictions
    predictions = [tree.predict(row) for row in test_data]

    # Calculate accuracy
    accuracy = accuracy_metric(test_y, predictions)
    
    # a[i] = accuracy 
    print(f"Accuracy: {accuracy:.2f}%")
    feature_names = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

    visualize_tree(tree,feature_names,class_names)
    # print(f"Average accuracy: {mean(a):.2f}%")
    plt.ioff()
    builtin_function(max_depth,1,train_x,train_y,test_x,test_y)

