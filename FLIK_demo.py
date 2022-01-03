import logging
import random
from concurrent.futures import ProcessPoolExecutor
from sklearn import metrics
from DecisionTree_demo import DecisionTreeClassifier
import os
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(1000000)
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class RandomForestClassifier(object):
        """
        :param  nb_trees:       Number of decision trees to use
        :param  nb_samples:     Number of samples to give to each tree
        :param  max_depth:      Maximum depth of the trees
        :param  max_workers:    Maximum number of processes to use for training
        """
        def __init__(self, nb_trees, nb_samples, max_depth=-1, max_workers=1):
            self.trees = []
            self.nb_trees = nb_trees
            self.nb_samples = nb_samples
            self.max_depth = max_depth
            self.real_max_depth = -1
            self.max_workers = max_workers

        """
        Trains self.nb_trees number of decision trees.
        :param  data:   A list of lists with the last element of each list being
                        the value to predict
        """
        def fit(self, data):
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                rand_fts = map(lambda x: [x, random.sample(data, self.nb_samples)],
                               range(self.nb_trees))

                self.trees = list(executor.map(self.train_tree, rand_fts))

        """
        Trains a single tree and returns it.
        :param  data:   A List containing the index of the tree being trained
                        and the data to train it
        """
        def train_tree(self, data):
            logging.info('Training tree {}'.format(data[0] + 1))
            tree = DecisionTreeClassifier(max_depth=self.max_depth,random_features=False,complete_random=True)
            tree.fit(data[1])

            return tree

        """
        Returns a prediction for the given feature. The result is the value that
        gets the most votes.
        :param  feature:    The features used to predict
        """
        def predict(self, feature):
            predictions = []

            for tree in self.trees:
                predictions.append(tree.predict(feature))

            # return max(set(predictions), key=predictions.count)
            return predictions


def make_fd_data(data_mnist, n, shuffle_flag=True):
    if shuffle_flag:
        random.shuffle(data_mnist)
    return [data_mnist[x::n] for x in range(n)]

def cal_inner(A,B):
    results=[]
    for i in range(0,len(B)):
        results.append(np.dot(A,B[i]))
    return results

def threshold_determination(Q):
    new_data = []
    for erer in Q:
        new_data=new_data+(np.ones(5)*erer).tolist()
    value_var = []
    Q_s=np.sort(new_data)
    for i in range(0,len(new_data)):
        L_l=Q_s[0:i]
        L_r = Q_s[i+1:len(Q_s)]
        value_var.append(abs(np.var(L_l) - np.var(L_r)))
    threshold_value=Q_s[np.where(value_var == np.min(value_var[1:len(value_var)-1]))]
    return threshold_value

def test_rf():
#####parameter#####
    NUM_CLIENTS = 5
    max_depthY = 500
    newclass = 3
    trainclass = [0, 1, 2]
    newclass_test = [0, 1, 2, newclass]
    nb_samples_t = 100
    # initial###
    model = []
    model_iF = []
#####data preparation####
    mean = np.array([3, 0])
    cov = np.eye(2)
    data_sy0=np.random.multivariate_normal(mean, cov, 1000)
    plt.scatter(data_sy0[:,0], data_sy0[:,1],c='red')
    data_sy0_label = np.ones(1000) * 0

    mean = np.array([8, 6])
    cov = np.diag([3,3])
    data_sy1 = np.random.multivariate_normal(mean, cov, 1000)
    plt.scatter(data_sy1[:, 0], data_sy1[:, 1], c='blue')
    data_sy1_label = np.ones(1000) * 1

    mean = np.array([-2, 10])
    cov = np.diag([2, 2])
    data_sy2 = np.random.multivariate_normal(mean, cov, 1000)
    plt.scatter(data_sy2[:, 0], data_sy2[:, 1], c='green')
    data_sy2_label = np.ones(1000) * 2

    mean = np.array([5, 15])
    cov = np.diag([2, 2])
    data_sy3 = np.random.multivariate_normal(mean, cov, 1000)
    plt.scatter(data_sy3[:, 0], data_sy3[:, 1], c='yellow')
    data_sy3_label = np.ones(1000) * 3

    plt.show()

    data_mnist=[]
    test_data = []
    test_label = []
    train_data=[]
    train_label=[]
#################### systhtic############
    images=np.vstack((data_sy0[0:800,:],data_sy1[0:800, :],data_sy2[0:800, :]))
    labels=np.concatenate((data_sy0_label[0:800],data_sy1_label[0:800],data_sy2_label[0:800]))


    images_test=np.vstack((data_sy0[801:1000,:],data_sy1[801:1000, :],data_sy2[801:1000, :],data_sy3[801:1000, :]))
    labels_test=np.concatenate((data_sy0_label[801:1000],data_sy1_label[801:1000],data_sy2_label[801:1000],data_sy3_label[801:1000]))

    for image, label in zip(images, labels):
        if label in trainclass:
            train_data.append(list(image))
            train_label.append(label)
            temp_1=list(image)
            temp_1.append(label)
            data_mnist.append(temp_1)

    for image_test, label_test in zip(images_test, labels_test):
        if label_test in newclass_test:
            test_data.append(list(image_test))
            test_label.append(label_test)

    data_mnist_fd = make_fd_data(data_mnist, n=NUM_CLIENTS)
    print('Data preparation is complete.')
    print('Training is begin.')
#####Training#####

    for i in range(0, NUM_CLIENTS):
        data_iso = np.array(data_mnist_fd[i])
        data_iso = np.delete(data_iso, -1, axis=1)
        rf = RandomForestClassifier(nb_trees=100, nb_samples=nb_samples_t, max_workers=1, max_depth=max_depthY)
        rf.fit(data_mnist_fd[i])
        model.append(rf)
        modelspe=[]
        ############caculate specification##########
        for image_train in data_iso:
            temp_2 = model[i].predict(image_train)
            partition_result = [k[3] for k in temp_2]
            specification_array = np.zeros(1000)
            for content in partition_result:
                temp_3=np.zeros(1000)
                temp_3[content]=1
                specification_array=specification_array+temp_3
            modelspe.append(specification_array)
        model[i].specification=modelspe
        print('Model %d finished' % i)
    print("initialization complete")

############Testing#######################
    specification_result = []
    final_label_results=[]
    for image_test, label_test in zip(test_data, test_label):
        aa_specification=[]
        aa_label=[]
        for i in range(0, NUM_CLIENTS):
            temp_4 = model[i].predict(image_test)
            partition_result = [k[3] for k in temp_4]
            label_result= [k[0] for k in temp_4]
            specification_array = np.zeros(1000)
            for content in partition_result:
                temp_5= np.zeros(1000)
                temp_5[content-1] = 1
                specification_array = specification_array + temp_5

            temp_6=cal_inner(specification_array, model[i].specification)

            aa_specification.append(np.average(temp_6))
            aa_label.append(max(label_result,key=label_result.count))
        specification_result.append(np.average(aa_specification))
        final_label_results.append(max(aa_label,key=aa_label.count))

    print("Testing complete")
    true_new_class_label=[0 if k==newclass else 1 for k in test_label]
    fpr, tpr, thresholds = metrics.roc_curve(true_new_class_label, specification_result, pos_label=1)
    result_AUC= metrics.auc(fpr, tpr)

    threshold_value=threshold_determination(specification_result)
    conbat_finale_label=[o if k>threshold_value else 3 for k,o in zip(specification_result,final_label_results)]

    result_ACC=metrics.accuracy_score(test_label, conbat_finale_label)

    print("AUC: %f" % result_AUC)
    print("ACC: %f" % result_ACC)

if __name__ == '__main__':
        print(os.cpu_count())
        logging.basicConfig(level=logging.INFO)
        test_rf()
