import re, string
import itertools

import nltk
from nltk.corpus import stopwords

# constants
BINARY = True
NONWORDS = re.compile('[\W_]+')
STOPWORDS = stopwords.words('english')

# read in a file
def scan(filename, exclude_stopwords = False, binary_label = False):
    data = []
    with open(filename, 'r') as f:
        while True:
            elements = {}

            for line in f:
                if line == '\n':
                    break
                try:
                    key, value = line.split(':', 1)
                    elements[key] = value
                except:
                    pass

            if not elements:
                break

            review = (elements['review/summary'] + ' ' + elements['review/text'])
            review = ' '.join(re.split(NONWORDS, review))
            review = review.strip().lower()

            if exclude_stopwords:
                review = ' '.join([w for w in review.split() if w not in STOPWORDS])

            score = float(elements['review/score'].strip())

            if binary_label:
                score = score_to_binary(score)

            datum = [review, score]
            data.append(datum)
    return data

def score_to_binary(score):
    if score >= 4:
        return 1
    else:
        return 0
