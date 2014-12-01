import math

# decision tree stuff
def entropy(reviews):
    prob = float(sum(reviews))/ len(reviews)
    return -1 * (prob * math.log(prob + 0.000001) + (1 - prob) * math.log((1 - prob + 0.00001)))

def information_gain(reviews, subreviews1, subreviews2):
    leftEntropy = entropy(subreviews2)
    leftEntropyWt = float(len(subreviews2))/len(reviews)
    rightEntropyWt = float(len(subreviews1))/len(reviews)
    rightEntropy = entropy(subreviews1)
    return entropy(reviews) - (leftEntropy * leftEntropyWt + rightEntropy * rightEntropyWt)

# natural language processing stuff
def freq(lst):
    freq = {}
    length = len(lst)
    for ele in lst:
        if ele not in freq:
            freq[ele] = 0
        freq[ele] += 1
    return (freq, length)

def get_unigram(review):
    return freq(review.split())

def get_unigram_list(review):
    return get_unigram(review)[0].keys()
