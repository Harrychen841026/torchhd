
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from pa2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, CategoricalNB
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, PC, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from pgmpy.base import DAG
from pgmpy.independencies import Independencies

from torchhd import functional
from torchhd import embeddings
import random

random.seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 40
IMG_SIZE = 28
NUM_LEVELS = 2
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.flatten = torch.nn.Flatten()

        self.position = embeddings.Random(size * size, DIMENSIONS)

        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        x = self.flatten(x)
        sample_hv = functional.bind(self.position.weight, self.value(x))
        sample_hv = functional.multiset(sample_hv)
        return functional.hard_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


model = Model(len(train_ds.classes), IMG_SIZE)
model = model.to(device)

clf = CategoricalNB()
training = torch.empty(size = (len(train_ds.classes), DIMENSIONS))
train_label = torch.empty(size = (len(train_ds.classes), 1))
with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)
        samples_hv = model.encode(samples)

        training = torch.vstack((training, samples_hv))
        train_label = torch.vstack((train_label, labels))

        model.classify.weight[labels] += samples_hv


    model.classify.weight[:] = F.normalize(model.classify.weight)
    training = training + 1
clf.fit(training[11:].tolist(), train_label[11:])
"""
finding best structure with HillClimbSearch algorithm
"""
training = training - 1
col = list(str(i) for i in range(DIMENSIONS + 1))
HC_training = pd.DataFrame(np.hstack((training[11:].tolist(), train_label[11:])), columns=col)
print(HC_training)

est = PC(HC_training)
best_model = est.estimate(return_type = "dag")

print(best_model.edges())
BN_model = BayesianNetwork()
for node in best_model.nodes() :
    BN_model.add_node(str(node))
for edge in best_model.edges() :
    BN_model.add_edge(edge[0], edge[1])
print(BN_model)
print(BN_model.nodes())
state_name = dict()
nodes = list(BN_model.nodes())
for i in nodes :
    if i == str(DIMENSIONS) :
        state_name.update({i: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
    else :
        state_name.update({i : [-1.0, 1.0]})
print(state_name)
int_nodes = [int(num) for num in nodes]
HC_training_trunc = HC_training.iloc[:, int_nodes]
print(HC_training_trunc)
BN_model.fit(HC_training_trunc, state_names =state_name , estimator=MaximumLikelihoodEstimator)

accuracy = torchmetrics.Accuracy()

testing = torch.empty(size = (len(test_ds.classes), DIMENSIONS))
test_label = torch.empty(size = (len(test_ds.classes), 1))

inference = VariableElimination(BN_model)
class_label = [str(DIMENSIONS)]
test_nodes = []
for node in nodes :
    if node != str(DIMENSIONS) :
        test_nodes.append(int(node))
corr = 0
with torch.no_grad():
    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)
        samples_hv = model.encode(samples)
        evid = dict()
        for i in test_nodes :
            evid.update({str(i): samples_hv[0,int(i)].tolist()})
        testing = torch.vstack((testing, samples_hv))
        test_label = torch.vstack((test_label, labels))
        phi_query = inference.map_query(variables = class_label, evidence = evid, show_progress = False)
        if phi_query[str(DIMENSIONS)] == labels :
            corr += 1
        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)
        accuracy.update(predictions.cpu(), labels)
    testing = testing + 1
print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
print("Bayes accuracy : ", corr / 10000)


prediction = []
pred = clf.predict_proba(testing[11:])

x, y = np.shape(pred)
for i in range(x):
    prediction.append(np.argmax(pred[i]))
corr = 0
for i in range(len(prediction)) :
    if prediction[i] == test_label[11+i] :
        corr += 1
print("NB accuracy : ", corr / len(prediction))


"""

generator = []
for i in range(1):
    generator = []
    for c in range(CLASS) :

        generator = gen(NBClassifier, i, c)
        #print(generator)
        #print(model.classify.weight.detach().size())
        #print(np.argmax(np.matmul(np.reshape(generator, (1, DIMENSIONS)), np.transpose(model.classify.weight.detach()))))
        #print(model.position.weight.size())
        dirty_values = np.multiply(model.position.weight, generator)
        pixel = []
        a, b = dirty_values.size()
        for j in range(a) :
            value = np.argmax(np.matmul(model.value.weight, np.transpose(dirty_values[j])))
            pixel.append(value/NUM_LEVELS)
        plt.subplot(1,10,c+1)
        plt.imshow(np.reshape(pixel, (IMG_SIZE,IMG_SIZE)))
    #plt.savefig("img_"+str(i)+".png")
    plt.savefig("img_2.png")


list_NB = []
for i in range(1) :
    print('Naive Bayes')
    accuracy, num_examples = evaluate(NBClassifier, i, train_subset=False)
    print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(accuracy, num_examples))
    list_NB.append(accuracy)
    print("mean and std of full :", (np.mean(list_NB), np.std(list_NB)))
"""