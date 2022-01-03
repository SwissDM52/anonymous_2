#!/usr/bin/env python3

import CSVReader
import random
from math import log, sqrt


class DecisionTreeClassifier:

    class DecisionNode:
        def __init__(self, col=-1, value=None, results=None, tb=None, fb=None, pathline=None,cumulation=None,current_max_depth = None,partition_id=None):
            self.col = col
            self.value = value
            self.pathline = pathline
            self.results = results
            self.tb = tb
            self.fb = fb
            self.current_max_depth = current_max_depth
            self.cumulation = cumulation
            self.partition_id = 0

    """
    :param  max_depth:          Maximum number of splits during training
    :param  random_features:    If False, all the features will be used to
                                train and predict. Otherwise, a random set of
                                size sqrt(nb features) will be chosen in the
                                features.
                                Usually, this option is used in a random
                                forest.
    """
    def __init__(self, max_depth=-1, random_features=False,complete_random=True):
        self.root_node = None
        self.max_depth = max_depth
        self.features_indexes = []
        self.current_max_depth = -1
        self.random_features = random_features
        self.complete_random=complete_random
        self.parttionid = 0

    """
    :param  rows:       The data used to rain the decision tree. It must be a
                        list of lists. The last vaue of each inner list is the
                        value to predict.
    :param  criterion:  The function used to split data at each node of the
                        tree. If None, the criterion used is entropy.
    """
    def fit(self, rows, criterion=None):
        if len(rows) < 1:
            raise ValueError("Not enough samples in the given dataset")

        if not criterion:
            criterion = self.entropy
        if self.random_features:
            self.features_indexes = self.choose_random_features(rows[0])
            rows = [self.get_features_subset(row) + [row[-1]] for row in rows]
        # self.parttionid = 0
        self.root_node = self.build_tree(rows, criterion, self.max_depth,self.complete_random,0)
        self.current_max_depth=self.root_node.current_max_depth

    """
    Returns a prediction for the given features.
    :param  features:   A list of values
    """
    def predict(self, features):
        if self.random_features:
            if not all(i in range(len(features))
                       for i in self.features_indexes):
                raise ValueError("The given features don't match\
                                 the training set")
            features = self.get_features_subset(features)

        return self.classify(features, self.root_node)

    """
    Randomly selects indexes in the given list.
    """
    def choose_random_features(self, row):
        nb_features = len(row) - 1
        return random.sample(range(nb_features), int(sqrt(nb_features)))

    """
    Returns the randomly selected values in the given features
    """
    def get_features_subset(self, row):
        return [row[i] for i in self.features_indexes]

    """
    Divides the given dataset depending on the value at the given column index.
    :param  rows:   The dataset
    :param  column: The index of the column used to split data
    :param  value:  The value used for the split
    """
    def divide_set(self, rows, column, value):
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value

        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]

        return set1, set2

    """
    Returns the occurence of each result in the given dataset.
    :param  rows:   A list of lists with the output at the last index of
                    each one
    """
    def unique_counts(self, rows):
        results = {}
        for row in rows:
            r = row[len(row) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    """
    Returns the entropy in the given rows.
    :param  rows:   A list of lists with the output at the last index of
                    each one
    """
    def entropy(self, rows):
        log2 = lambda x: log(x) / log(2)
        results = self.unique_counts(rows)
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            ent = ent - p * log2(p)
        return ent

    """
    Recursively creates the decision tree by splitting the dataset until no
    gain of information is added, or until the max depth is reached.
    :param  rows:   The dataset
    :param  func:   The function used to calculate the best split and stop
                    condition
    :param  depth:  The current depth in the tree
    """
    def build_tree(self, rows, func, depth,complete_random,accum):

            if len(rows) == 0:
                 self.current_max_depth = -1
                 self.parttionid =self.parttionid  +1
                 return self.DecisionNode(pathline = depth,
                                          cumulation = self.parttionid,
                                          current_max_depth = depth)
            if depth == 0:
                self.current_max_depth = -1
                self.parttionid =self.parttionid  +1
                return self.DecisionNode(results = self.unique_counts(rows),
                                        pathline = depth,
                                        cumulation = self.parttionid,
                                        current_max_depth = depth)

            if len(rows) > 1:
                # while 1:
                col_random = random.choice(range(0, len(rows[0]) - 1))
                column_values = [x[col_random] for x in rows]
                max_temp=max(column_values)
                min_temp=min(column_values)

                values_random= random.uniform(min_temp,max_temp)
                set1, set2 = self.divide_set(rows, col_random, values_random)
                best_sets = (set1, set2)

                trueBranch = self.build_tree(best_sets[0], func, depth - 1, complete_random,accum)
                falseBranch = self.build_tree(best_sets[1], func, depth - 1, complete_random,accum)
                return self.DecisionNode(col=col_random,
                                         value=values_random,
                                         tb=trueBranch,
                                         fb=falseBranch,
                                         results=self.unique_counts(rows),
                                         pathline=depth,
                                         cumulation=accum,
                                         current_max_depth=min([depth,trueBranch.current_max_depth,falseBranch.current_max_depth])
                )
            else:
                self.parttionid = self.parttionid +1
                return self.DecisionNode(results=self.unique_counts(rows),
                                         pathline=depth,
                                         cumulation=self.parttionid,
                                         current_max_depth=depth)

    """
    Makes a prediction using the given features.
    :param  observation:    The features to use to predict
    :param  tree:           The current node
    """
    def classify(self, observation, tree):
        if tree.results is not None and tree.tb is None and tree.fb is None :
            return [list(tree.results.keys())[0],
                    tree.pathline,
                    sum(list(tree.results.values())),
                    tree.cumulation
                    ]
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            if branch.col == -1 and branch.results is None:
                return [list(tree.results.keys())[0],
                        tree.pathline,
                        sum(list(tree.results.values())),
                        branch.cumulation]
            else:
               return self.classify(observation, branch)



def test_tree():
    data = CSVReader.read_csv("../data/income.csv")
    tree = DecisionTreeClassifier()
    tree.fit(data)

    print(tree.predict([39, 'State-gov', 'Bachelors', 13, 'Never-married',
                        'Adm-clerical', 'Not-in-family', 'White', 'Male',
                        2174, 0, 40, 'United-States']))


if __name__ == '__main__':
    test_tree()
