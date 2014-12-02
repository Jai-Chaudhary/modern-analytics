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
log_sum_of_every_word_absence_per_decade = {}

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
                                 "year": int(year),
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
    global log_sum_of_every_word_absence_per_decade
    for decade, movies in movies_per_decade.iteritems():
        word_freq_over_movies = collections.defaultdict(dict)
        for movie in movies:
            try:
                words = collections.Counter(list(itertools.chain(*[word_tokenize(t) for t in sent_tokenize(movie['summary'].lower())])))
                for (word, count) in words.iteritems():
                    word_freq_over_movies[word][count] = word_freq_over_movies[word].get(count, 0) + 1 
            except:
                print movie['title']
        prob_of_word_given_decade[decade] = {}
        log_sum_of_every_word_absence_per_decade[decade] = 0
        for (word, pmf) in word_freq_over_movies.iteritems():
            prob_of_word_given_decade[decade][word] = {}
            for count in pmf:
                prob_of_word_given_decade[decade][word][count] = float(word_freq_over_movies[word][count])/ len(movies) 
            word_absence_prob = 1 - sum(prob_of_word_given_decade[decade][word].values())
            prob_of_word_given_decade[decade][word][0] = 0.00001 if word_absence_prob == 0 else word_absence_prob
            log_sum_of_every_word_absence_per_decade[decade] += math.log(prob_of_word_given_decade[decade][word][0])

def test_naive_bayes(movie):
    global prob_of_word_given_decade
    global log_sum_of_every_word_absence_per_decade

    log_posterior = {'1930': -1000000, '1940': -1000000, '1950': -1000000, '1960': -1000000, '1970': -1000000, '1980': -1000000, '1990': -1000000, '2000': -1000000, '2010': -1000000}

    try:
        test_movie_word_freq = collections.Counter(list(itertools.chain(*[word_tokenize(t) for t in sent_tokenize(movie['summary'].lower())])))

        for decade in prob_of_word_given_decade.keys():
            log_posterior[decade] = 0
            for word in test_movie_word_freq.keys():
                if prob_of_word_given_decade[decade].get(word, -1) != -1:
                    if prob_of_word_given_decade[decade][word].get(test_movie_word_freq[word], -1) != -1:
                        log_posterior[decade] += math.log(prob_of_word_given_decade[decade][word][test_movie_word_freq[word]])
                        log_posterior[decade] -= math.log(prob_of_word_given_decade[decade][word][0])
                    else:
                        log_posterior[decade] += -5
                else:
                    log_posterior[decade] += -5
    except:
        print movie['title']
    return log_posterior

def plot_posterior_histogram(test_movies):
    for movie in test_movies:
        posterior = test_naive_bayes(movie)

        normalizer = sum(posterior.values())

        pmf = {}
        for decade, post_prob in posterior.iteritems():
            pmf[decade] = (float(post_prob) / normalizer)

        pmf = collections.OrderedDict(sorted(pmf.items(), key=lambda x: int(x[0])))
        index = np.arange(len(pmf))
        print pmf

        rects1 = plt.bar(index, pmf.values(), 0.35)

        plt.xlabel('Decade')
        plt.ylabel('Probability')
        plt.title('Posterior Probability for each Decade for ' + movie['title'])
        plt.xticks(index + 0.35, pmf.keys())

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    all_movies = list(load_all_movies("plot.list.gz"))
    movies_per_decade = {'1930':[], '1940':[], '1950':[], '1960':[], '1970':[], '1980':[], '1990':[], '2000':[], '2010':[]}
    test_movies_to_plot = []

    for current_movie in all_movies:
        movies_per_decade[str(10*int(int(current_movie['year'])/10))].append(current_movie)
        if 'Finding Nemo' == current_movie['title']:
            test_movies_to_plot.append(current_movie)
        elif 'The Matrix' == current_movie['title']:
            test_movies_to_plot.append(current_movie)
        elif 'Gone with the Wind' == current_movie['title']:
            test_movies_to_plot.append(current_movie)
        elif 'Harry Potter and the Goblet of Fire' == current_movie['title']:
            test_movies_to_plot.append(current_movie)
        elif 'Avatar' == current_movie['title'] and current_movie['year'] == 2009:
            test_movies_to_plot.append(current_movie)

    sampled_movies_per_decade = {decade: random.sample(movies, 6000) for (decade, movies) in movies_per_decade.iteritems()}
    train_movies_per_decade_sample = {decade: [movies[i] for i in range(len(movies)) if i % 2 == 0] for (decade, movies) in sampled_movies_per_decade.iteritems()}
    test_movies_per_decade_sample = {decade: [movies[i] for i in range(len(movies)) if i % 2 != 0] for (decade, movies) in sampled_movies_per_decade.iteritems()}

    # sampled_movies =  reduce(lambda x, y: x+y, sampled_movies_per_decade.values())
    # plot_count_per_decade()
    # plot_count_per_decade_given_word("zombie", sampled_movies)
    train_naive_bayes_model(train_movies_per_decade_sample)
    true_positive = {'1930': 0, '1940': 0, '1950': 0, '1960': 0, '1970': 0, '1980': 0, '1990': 0, '2000': 0, '2010': 0}
    
    for decade, movies in test_movies_per_decade_sample.iteritems():
        for movie in movies:
            log_posterior = test_naive_bayes(movie)
            if max(log_posterior.iteritems(), key=operator.itemgetter(1))[0] == decade:
                true_positive[decade] += 1
    print true_positive

    # plot_posterior_histogram(test_movies_to_plot)
    # len(all_movies)
    # => 379451