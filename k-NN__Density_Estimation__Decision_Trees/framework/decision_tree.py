import scan
import utils
import numpy as np
from collections import defaultdict

class DecisionTree:
    node_label = None  # takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
    node_word = None
    left = None
    right = None

    def decision(self, data):
        if (self.node_word in data):
            self = self.left
        else:
            self = self.right
        return self.go(data)

    def go(self, data):
        if self.node_label != None:
            return self.node_label
        return self.decision(data)

# http://en.wikipedia.org/wiki/ID3_algorithm
def train(data):
    attributeList = mostPopularPositiveNegative(data)
    print "Training..."
    return create_tree(data, attributeList, 0)

def create_tree(data, attributeList, depth):
    print "Pre-order Traversal Depth:", depth, "NumberOfDataPoints", len(data)
    reviews = [ review[1] for review in data]
    sumOfReviews = sum(reviews)


    if depth == 19 or attributeList == None:
        leaf = DecisionTree()
        leaf.node_label = 1 if sumOfReviews > len(reviews) / 2 else 0

        return leaf

    if sumOfReviews == len(reviews):
        leaf = DecisionTree()
        leaf.node_label = 1

        return leaf 

    if sumOfReviews == 0:
        leaf = DecisionTree()
        leaf.node_label = 0

        return leaf 

    maxInfoGain = 0
    maxInfoGainIndex = -1
    for i, attribute in enumerate(attributeList):
        withAttribute = []
        withOutAttribute = []
        infoGain = 0
        for datum in data:
            if attribute in datum[0]:
                withAttribute.append(datum[1])
            else:
                withOutAttribute.append(datum[1])
        if withAttribute == [] or withOutAttribute == []:
            infoGain = 0
        else:
            infoGain = utils.information_gain(reviews, withAttribute, withOutAttribute)
        if maxInfoGain < infoGain:
            maxInfoGain = infoGain
            maxInfoGainIndex = i
    leftChildReviews = []
    rightChildReviews = []
    for review in data:
        if attributeList[maxInfoGainIndex] in review[0]:
            leftChildReviews.append(review)
        else:
            rightChildReviews.append(review)

    node = DecisionTree()
    node.node_word = attributeList[maxInfoGainIndex]
    attributeList.remove(attributeList[maxInfoGainIndex])
    node.left = create_tree(leftChildReviews, attributeList, depth + 1)
    node.right = create_tree(rightChildReviews, attributeList, depth + 1)

    return node

def test(decision_tree, data):
    truePositiveNegative = 0
    for each_review in data:
        score = decision_tree.go(each_review[0])
        if score == each_review[1]:
            truePositiveNegative += 1
    print "Accuracy: ", float(truePositiveNegative) / len(data)


def mostPopularPositiveNegative(data, n = 500):
    allUnigrams = defaultdict(int)
    positiveUnigrams = defaultdict(int)
    negativeUnigrams = defaultdict(int)
    for review in data:
        for unigram, freq in utils.get_unigram(review[0])[0].items():
            allUnigrams[unigram] += freq
            positiveUnigrams[unigram] += freq if review[1] == 1 else 0
            negativeUnigrams[unigram] += freq if review[1] == 0 else 0

    topAllUnigrams = sorted(allUnigrams, key=allUnigrams.get, reverse=True)
    topPositiveUnigrams = sorted(positiveUnigrams, key=positiveUnigrams.get, reverse=True)
    topNegativeUnigrams = sorted(negativeUnigrams, key=negativeUnigrams.get, reverse=True)
    print ("Top 30 Popular Unigrams", "Top 30 Positive Unigrams", "Top 30 Negative Unigrams")
    for i in xrange(30):
        print repr(topAllUnigrams[i]).center(30), repr(topPositiveUnigrams[i]).center(30), repr(topNegativeUnigrams[i]).center(30)

    return topPositiveUnigrams + topNegativeUnigrams
