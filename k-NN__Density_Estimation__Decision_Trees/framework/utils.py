# decision tree stuff
def entropy():
    raise 'not implemented'

def information_gain():
    raise 'not implemented'

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
