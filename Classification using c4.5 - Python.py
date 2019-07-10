from collections import Counter
import numpy as np
import re


# In[17]:


class Node(object):
    """Decision tree nodes.

    Attributes:
        label: either a feature name (internal node),
            or a class label (leaf).
        branches (list): of tuples, each is a
            (branch, node) object pair.

    Note:
        __str()__ is called implicitly by print(node).

    """

    def __init__(self, label):
        self.label = label
        self.branches = []

    def __str__(self, level=0, branch_label=''):
        """Print, recursivley, the subtree whose root is the node.

        The labels of children will be indented, and the labels
        of their children will be indented further.

        """
        ret = "     " * level + str(branch_label) + ':' + str(self.label) + "\n"
        for branch in self.branches:
            ret += branch[1].__str__(level + 1, branch[0].label)
        return ret


# In[18]:


class Branch(object):
    """Decision tree edges.

    Each branch is an edge between a parent node
    (labeled by a feature name) and a child node
    (labeled by a feature, or a class name).

    Attributes:
        value: a value of the feature whose name is the label
            of the parent node
        cmp (int): an int value that determines the behavior
            of is_match().
        label (str): label of the edge in the tree, used
            when printing the tree.

    """

    def __init__(self, value, cmp: int = 0):
        self.value = value
        self.cmp = cmp
        if cmp == -1:
            self.label = "<=" + str(value)
        elif cmp == 1:
            self.label = ">" + str(value)
        else:
            self.label = str(value)

    def is_match(self, other_value):
        """Compare other_value to self.value.

        Returns:
            True, if:
                other_value == self.value (when self.cmp is 0)
                other_value <= self.value (when self.cmp is -1)
                other_value >  self.value (when self.cmp is 1)
            False otherwise.

        """
        if self.cmp == -1:
            return other_value <= self.value
        elif self.cmp == 1:
            return other_value > self.value
        else:
            return other_value == self.value


# In[19]:


def entropy_dataset(dataset: list) -> float:
    """Calculate the entropy of the dataset.

    Notes:
        Requires numpy.

    Args:
        dataset: of samples.

    Returns:
        entropy

    """
    labels = [sample[-1] for sample in dataset]
    return entropy_values(labels)


# In[20]:


def entropy_values(values: list) -> float:
    """Calculate the entropy of the input values."""
    if len(values) == 0:  # entropy is undefined?
        return 0

    counter = Counter(values)
    frequencies = np.array(list(counter.values()))
    probabilities = frequencies / len(values)
    entropy = -1 * np.dot(probabilities,
                          np.log2(probabilities))
    return entropy


# In[21]:


def most_frequent(values):
    """Return the most frequent value among input values."""
    counts = Counter(values)
    return counts.most_common(1)[0][0]


# In[22]:


def gainratio_nominal(dataset: list, index) -> float:
    """Compute gainratio of the feature given by the index.

    Formula:
        gainratio =
            information_gain(dataset, feature) / entropy(feature)
    where
        information_gain(dataset, feature) =
            entropy(dataset) - entropy(children:dataset-split-by-feature)
        entropy(feature): entropy computed w.r.t. feature values.

    """
    feature_values = [sample[index] for sample in dataset]
    entropy_feature = entropy_values(feature_values)
    feature_values = set(feature_values)

    entropy_parent = entropy_dataset(dataset)
    entropy_children = 0
    for value in feature_values:
        labels_child = [sample[-1] for sample in dataset if sample[index] == value]
        probability_child = len(labels_child) / len(dataset)
        entropy_children += probability_child * entropy_values(labels_child)
    gain = entropy_parent - entropy_children
    gainratio = gain / entropy_feature
    return gainratio


# In[23]:


def gainratio_numeric(dataset: list, index) -> float:
    feature_values = [sample[index] for sample in dataset]
    feature_values = set(feature_values)
    entropy_feature = entropy_values(feature_values)
    entropy_parent = entropy_dataset(dataset)
    entropy_child1 = 0
    entropy_child2 = 0
    for value in feature_values:
        child = [sample[-1] for sample in dataset if sample[index] == value]
        child1 = [sample[-1] for sample in dataset if sample[index] <= value]
        child2 = [sample[-1] for sample in dataset if sample[index] > value]
        probability_child1 = len(child1)/len(dataset)
        probability_child2 = len(child2)/len(dataset)
        entropy_child1 += probability_child1 * entropy_values(child1)
        gain = entropy_parent - entropy_child1
        gainratio = gain / entropy_feature
        entropy_child2 += probability_child2 * entropy_values(child2)
        gain_1 = entropy_parent - entropy_child2
        gainratio_1 = gain_1/ entropy_feature
        splitted = []
        splitted = [child1,child2]
        if gainratio > gainratio_1:
            return gainratio,splitted
        else:
            return gainratio_1,splitted



# In[24]:


def index_bestratio(dataset: list, isnumeric: list):
    """Find best feature to split dataset.

    Args:
        dataset: list of samples
        isnumeric: bool values, if each feature is numeric?

    Returns:
        index_best (int): index of feature that has the best
            gainratio.
        best_value (numeric): returned only if index_best is
            the index of a numeric feature.

    """
    number_features = len(dataset[0]) - 1

    best_ratio = 0;
    index_best = -1;
    for index in range(number_features):
        if isnumeric[index]:
            # partition values
            pvalues = set([row[index] for row in dataset])
            for p in pvalues:
                # copy dataset
                workset = [s[:] for s in dataset]
                for i in range(len(workset)):
                    workset[i][index] = (workset[i][index] > p)
                gainratio = gainratio_nominal(workset, index)
                if gainratio > best_ratio:
                    best_ratio = gainratio
                    index_best = index
                    best_value = p
                    # print('best {} {}'.format(index_best,best_value))
        else:
            gainratio = gainratio_nominal(dataset, index)
            if gainratio > best_ratio:
                best_ratio = gainratio
                index_best = index
                best_value = None

    """Change the loop above to make use of isnumeric;
    to find the feature that gives the highest ratio
    among all features whether nomial or numeric.

    """

    return index_best, best_value


# In[25]:


def create_tree(dataset, feature_names, isnumeric: list = None):
    feature_names_ = feature_names[:]
    class_labels = [sample[-1] for sample in dataset]

    if class_labels.count(class_labels[0]) == len(class_labels):
        leafnode = Node(class_labels[0])
        return leafnode
    if len(dataset[0]) == 1:
        leafnode = Node(most_frequent(class_labels))
        return leafnode

    if isnumeric is None:
        isnumeric = []
        for index in range(len(feature_names)):
            if type(dataset[0][index]).__name__ in ['int', 'float']:
                isnumeric.append(True)
            else:
                isnumeric.append(False)
    isnumeric_ = isnumeric[:]

    index_bestfeature, bestvalue = index_bestratio(dataset, isnumeric)
    if bestvalue != None:
        print('here ' + feature_names_[index_bestfeature])
        for i in range(len(dataset)):
            if (dataset[i][index_bestfeature] > bestvalue):
                dataset[i][index_bestfeature] = '>' + str(bestvalue)
            else:
                dataset[i][index_bestfeature] = '<=' + str(bestvalue)

    root = Node(feature_names_[index_bestfeature])
    del (feature_names_[index_bestfeature])

    # nominal
    values = set([sample[index_bestfeature] for sample in dataset])
    for value in values:
        feature_names__ = feature_names_[:]
        isnumeric__ = isnumeric_[:]
        child = []
        for sample in dataset:
            if sample[index_bestfeature] == value:
                sample_ = sample[:index_bestfeature] + sample[index_bestfeature + 1:]
                child.append(sample_)
        branch = Branch(value)
        root.branches.append((branch, create_tree(child, feature_names__, None)))

    return root


# In[26]:


def classify(root, feature_names, new_sample):
    if len(root.branches) == 0:
        guess = root.label
    else:
        index = feature_names.index(root.label)

        for branch in root.branches:
            if branch[0].is_match(new_sample[index]):
                guess = classify(branch[1], feature_names, new_sample)
                break

    return guess


# In[27]:


def create_fish_dataset():
    datamatrix = [[1, 1, 'yes'],
                  [1, 1, 'yes'],
                  [1, 0, 'no'],
                  [0, 1, 'no'],
                  [0, 1, 'no']]
    feature_names = ['no surfacing', 'flippets']
    return datamatrix, feature_names


# In[28]:


def load_lenses(filename):
    fr = open(filename, 'r')
    datamatrix = [line.strip().split('\t') for line in fr.readlines()]
    fr.close()
    feature_names = ['age', 'prescript', 'astigmatic', 'tearrate']
    return datamatrix, feature_names


# In[29]:


def load_golf(filename):
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    datamatrix = []
    for line in lines:
        tokens = re.split('\W+', line.strip())
        sample = [tokens[0]]
        sample.extend(list(map(int, tokens[1:3])))
        sample.append(tokens[3])
        if len(tokens) == 5:
            sample.append(tokens[4])
        else:
            sample.append("Don't Play")
        datamatrix.append(sample)
    feature_names = ['outlook', 'temperature', 'humidity', 'windy']
    isnumeric = [False, True, True, False]
    return datamatrix, feature_names, isnumeric


# In[33]:


if __name__ == "__main__":

    print("Golf dataset:")
    golf_data, golf_featurenames, golf_isnumeric = load_golf ("golf.data")
    for sample in golf_data:
        print(sample)
    golf_root = create_tree(golf_data, golf_featurenames)
    print("golf: tree")
    print(golf_root)