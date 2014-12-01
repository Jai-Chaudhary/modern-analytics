import gzip
import re
import matplotlib.pyplot as plt
import operator
import numpy as np
import collections
import random
import math
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import itertools

prob_of_word_given_decade = {}
all_words_in_all_movies = []

def load_all_movies(filename):
    """
    Load and parse 'plot.list.gz'. Yields each consecutive movie as a dictionary:
        {"title": "The movie's title",
         "year": The decade of the movie, like 1950 or 1980
         "identifier": Full key of IMDB's text string,
         "summary": "The movie's plot summary"
        }
    You can download `plot.list.gz` from http://www.imdb.com/interfaces
    """
    assert "plot.list.gz" in filename # Or whatever you called it
    current_movie = None
    movie_regexp = re.compile("MV: ((.*?) \(([0-9]+).*\)(.*))")
    skipped = 0
    for line in gzip.open(filename):
        if line.startswith("MV"):
            if current_movie:
                # Fix up description and send it on
                current_movie['summary'] = "\n".join(current_movie['summary'] )
                yield current_movie
            current_movie = None
            try:
                identifier, title, year, episode = movie_regexp.match(line).groups()
                if int(year) < 1930 or int(year) > 2014:
                    # Something went wrong here
                    raise ValueError(identifier)
                current_movie = {"title": title,
                                 "year": 10*int(int(year)/10),
                                 'identifier': identifier,
                                 'episode': episode,
                                 "summary": []}
            except:
                skipped += 1
        if line.startswith("PL: ") and current_movie:
            # Add to the current movie's description
            current_movie['summary'].append(line.replace("PL: ",""))
            
    print "Skipped",skipped

def plot_count_per_decade():

    count_per_decade = {'1930':0, '1940':0, '1950':0, '1960':0, '1970':0, '1980':0, '1990':0, '2000':0, '2010':0}
    for current_movie in load_all_movies("plot.list.gz"):
        count_per_decade[str(10*int(int(current_movie['year'])/10))] += 1

    total_movies = sum(count_per_decade.values())
    
    pmf = {}
    for movie, movie_count in count_per_decade.iteritems():
        pmf[movie] = (float(movie_count) / total_movies)

    pmf = collections.OrderedDict(sorted(pmf.items(), key=lambda x: int(x[0])))
    index = np.arange(len(pmf))
    print pmf

    rects1 = plt.bar(index, pmf.values(), 0.35)

    plt.xlabel('Decade')
    plt.ylabel('Probability')
    plt.title('PMF of P(Y)')
    plt.xticks(index + 0.35, pmf.keys())

    plt.tight_layout()
    plt.show()

def plot_count_per_decade_given_word(word, movies):

    count_per_decade = {'1930':0, '1940':0, '1950':0, '1960':0, '1970':0, '1980':0, '1990':0, '2000':0, '2010':0}
    
    for current_movie in movies:
        if current_movie['summary'].find(word) != -1:
            count_per_decade[str(10*int(int(current_movie['year'])/10))] += 1

    total_movies = sum(count_per_decade.values())
    print total_movies

    pmf = {}
    for movie, movie_count in count_per_decade.iteritems():
        pmf[movie] = (float(movie_count) / total_movies)

    pmf = collections.OrderedDict(sorted(pmf.items(), key=lambda x: int(x[0])))
    index = np.arange(len(pmf))
    print pmf

    rects1 = plt.bar(index, pmf.values(), 0.35)

    plt.xlabel('Decade')
    plt.ylabel('Probability')
    plt.title('PMF of P(Y) given ' + word)
    plt.xticks(index + 0.35, pmf.keys())

    plt.tight_layout()
    plt.show()

def train_naive_bayes_model(movies_per_decade):

    global prob_of_word_given_decade
    global all_words_in_all_movies
    for decade, movies in movies_per_decade.iteritems():
        movie_bag_of_words = collections.defaultdict(dict)
        for movie in movies:
            try:
                words = collections.Counter(list(itertools.chain(*[word_tokenize(t) for t in sent_tokenize(movie['summary'].lower())])))
                all_words_in_all_movies += words.keys()
                for (word, count) in words.iteritems():
                    movie_bag_of_words[word][count] = movie_bag_of_words[word].get(count, 0) + 1 
            except:
                print movie
        prob_of_word_given_decade[decade] = {}
        for (word, pmf) in movie_bag_of_words.iteritems():
            prob_of_word_given_decade[decade][word] = {}
            for count in pmf:
                prob_of_word_given_decade[decade][word][count] = float(movie_bag_of_words[word][count])/ len(movies) 
            prob_of_word_given_decade[decade][word][0] = 1 - sum(prob_of_word_given_decade[decade][word].values())
    

    print prob_of_word_given_decade

def test_naive_bayes(movie):
    global prob_of_word_given_decade
    global all_words_in_all_movies

    test_movie_word_freq = collections.Counter(list(itertools.chain(*[word_tokenize(t) for t in sent_tokenize(movie['summary'].lower())])))
    
    posterior = {}
    for decade in prob_of_word_given_decade.keys():
        posterior[decade] = 0
        for word in all_words_in_all_movies:
            if prob_of_word_given_decade[decade].get(word, -1) != -1:
                if (test_movie_word_freq.get(word, -1) != -1):
                    if prob_of_word_given_decade[decade][word].get(test_movie_word_freq[word], -1) != -1:
                        if prob_of_word_given_decade[decade][word][test_movie_word_freq[word]] <= 0:
                            print prob_of_word_given_decade[decade][word][test_movie_word_freq[word]]
                        posterior[decade] += math.log(prob_of_word_given_decade[decade][word][test_movie_word_freq[word]])
                    else:
                        posterior[decade] += -5
                else:
                    if prob_of_word_given_decade[decade][word][test_movie_word_freq[word]] <= 0:
                        print prob_of_word_given_decade[decade][word][test_movie_word_freq[word]]
                    posterior[decade] += math.log(prob_of_word_given_decade[decade][word][0])
            else:
                posterior[decade] += -5
    print posterior



if __name__ == '__main__':
    all_movies = list(load_all_movies("plot.list.gz"))
    count = {'a' : 0, 'r': 0}

    movies_per_decade = {'1930':[], '1940':[], '1950':[], '1960':[], '1970':[], '1980':[], '1990':[], '2000':[], '2010':[]}

    for current_movie in all_movies:
        movies_per_decade[str(10*int(int(current_movie['year'])/10))].append(current_movie)

    sampled_movies_per_decade = {decade: random.sample(movies, 10) for (decade, movies) in movies_per_decade.iteritems()}
    # print sampled_movies_per_decade.values()
    
    sampled_movies =  reduce(lambda x, y: x+y, sampled_movies_per_decade.values())
    # # plot_count_per_decade()
    # plot_count_per_decade_given_word("zombie", sampled_movies)
    train_naive_bayes_model(sampled_movies_per_decade)
    # test_naive_bayes(sampled_movies_per_decade['2000'][0])

    # len(all_movies)
    # => 379451