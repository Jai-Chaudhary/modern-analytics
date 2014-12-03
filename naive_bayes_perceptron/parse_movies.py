import gzip
import re
import matplotlib.pyplot as plt
import operator
import numpy as np
import collections
import random
import math
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

import itertools
import string
import pprint
import pylab as pl
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

stop = stopwords.words('english')
exclude = set(string.punctuation)

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
                # words = collections.Counter(list(itertools.chain(*[word_tokenize(t) for t in sent_tokenize(movie['summary'].lower())])))
                filteredStr = ''.join(ch for ch in movie['summary'].lower() if ch not in exclude)
                relevant_term_freq = collections.Counter([term for term in filteredStr.split() if term not in stop and len(term) > 2])
                
                for (word, count) in relevant_term_freq.iteritems():
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

def test_naive_bayes(movie, exclude_word_list_per_decade):
    global prob_of_word_given_decade
    global log_sum_of_every_word_absence_per_decade
    
    log_posterior = {}

    # test_movie_word_freq = collections.Counter(list(itertools.chain(*[word_tokenize(t) for t in sent_tokenize(movie['summary'].lower())])))
    filteredStr = ''.join(ch for ch in movie['summary'].lower() if ch not in exclude)
    test_movie_word_freq = collections.Counter([term for term in filteredStr.split() if term not in stop and len(term) > 2])

    for decade in prob_of_word_given_decade.keys():
        log_posterior[decade] = log_sum_of_every_word_absence_per_decade[decade]
        for word in test_movie_word_freq.keys():
            if prob_of_word_given_decade[decade].get(word, -1) != -1 and word not in exclude_word_list_per_decade[decade]:
                # if prob_of_word_given_decade[decade][word].get(test_movie_word_freq[word], -1) != -1:
                    # log_posterior[decade] += math.log(prob_of_word_given_decade[decade][word][test_movie_word_freq[word]])
                    # log_posterior[decade] -= math.log(prob_of_word_given_decade[decade][word][0])
                # else:
                #     print "Word not found with count"
                #     print word, test_movie_word_freq[word]
                #     log_posterior[decade] += -5
                log_posterior[decade] += math.log(1 - prob_of_word_given_decade[decade][word][0])
                log_posterior[decade] -= math.log(prob_of_word_given_decade[decade][word][0])
            else:
                # print "Word not found", word
                log_posterior[decade] += -5


    return log_posterior

def plot_posterior_histogram(test_movies):
    for movie in test_movies:
        posterior = test_naive_bayes(movie, [])

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

def plot_cumulative_match_curve(train_movies_per_decade_sample, test_movies_per_decade_sample):
    train_naive_bayes_model(train_movies_per_decade_sample)

    cumulative_match = {}
    for k in range(1,10):
        cumulative_match[k] = 0
        for decade, movies in test_movies_per_decade_sample.iteritems():
            for movie in movies:
                log_posterior = test_naive_bayes(movie, [])
                if decade in dict(sorted(log_posterior.iteritems(), key=operator.itemgetter(1), reverse=True)[:k]).keys(): 
                    cumulative_match[k] += 1
    print cumulative_match

def plot_confusion_matrix(train_movies_per_decade_sample, test_movies_per_decade_sample, exclude_word_list_per_decade):
    train_naive_bayes_model(train_movies_per_decade_sample)

    confusion_matrix = {}
    for i, j in zip(range(1,10), range(1,10)):
        for decade, movies in test_movies_per_decade_sample.iteritems():
            confusion_matrix[decade] = {}
            for movie in movies:
                log_posterior = test_naive_bayes(movie, exclude_word_list_per_decade)
                # confusion_matrix[actual_decade][predicted_decade]
                predicted_decade = sorted(log_posterior.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]
                confusion_matrix[decade][predicted_decade] = confusion_matrix[decade].get(predicted_decade, 0) + 1
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(confusion_matrix)

    pl.pcolor(np.array([[ confusion_matrix[actual_decade][decade] for decade in predicted_decades] for (actual_decade, predicted_decades) in confusion_matrix.iteritems()]))
    pl.colorbar()
    pl.show()


def top_n_info_words(sampled_movies_per_decade, n):
    word_presence_prob = collections.defaultdict(dict)
    for decade, movies in sampled_movies_per_decade.iteritems():
        word_freq_over_movies = collections.defaultdict(dict)
        for movie in movies:
            # words = collections.Counter(list(itertools.chain(*[word_tokenize(t) for t in sent_tokenize(movie['summary'].lower())])))
            filteredStr = ''.join(ch for ch in movie['summary'].lower() if ch not in exclude)
            relevant_term_freq = collections.Counter([term for term in filteredStr.split() if term not in stop and len(term) > 2])
            
            for (word, count) in relevant_term_freq.iteritems():
                word_freq_over_movies[word][count] = word_freq_over_movies[word].get(count, 0) + 1 

        prob_of_word_given_decade[decade] = {}
        log_sum_of_every_word_absence_per_decade[decade] = 0

        for (word, pmf) in word_freq_over_movies.iteritems():
            prob_of_word_given_decade[decade][word] = {}
            for count in pmf:
                prob_of_word_given_decade[decade][word][count] = float(word_freq_over_movies[word][count])/ len(movies) 
            word_presence_prob[word][decade] = sum(prob_of_word_given_decade[decade][word].values())

    info_words_per_decade = collections.defaultdict(dict)    
    for (word, decades) in word_presence_prob.iteritems():
        for decade in decades.keys():
            info_words_per_decade[decade][word] = word_presence_prob[word][decade] / min(word_presence_prob[word].values())
    
    top_n_info_words = {}
    for (decade, words) in info_words_per_decade.iteritems():    
        top_n_info_words[decade] = sorted(words.iteritems(), key=operator.itemgetter(1), reverse=True)[:n]

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(top_n_info_words)

    return top_n_info_words

def confusion_matrix_strip_top_100_informative_words(sampled_movies_per_decade, train_movies_per_decade_sample, test_movies_per_decade_sample):
    exclude_word_list_per_decade = top_n_info_words(sampled_movies_per_decade, 100)
    
    train_naive_bayes_model(train_movies_per_decade_sample)

    plot_confusion_matrix(train_movies_per_decade_sample, test_movies_per_decade_sample, exclude_word_list_per_decade)


def sklearn_naive_bayes(train_movies_sample, test_movies_sample):
    count_vect = CountVectorizer()
    gnb = GaussianNB()
    
    train_movies = {movie['summary'].decode('ascii', 'ignore').encode('utf-8'): str(10*int(int(movie['year'])/10)) for movie in train_movies_sample}
    train_movie_data = count_vect.fit_transform(train_movies.keys()).toarray()
    train_movie_target = train_movies.values()
    print len(train_movie_data), len(train_movie_target)

    test_movies = {movie['summary'].decode('ascii', 'ignore').encode('utf-8'): str(10*int(int(movie['year'])/10)) for movie in test_movies_sample}
    test_movie_data = count_vect.transform(test_movies.keys()).toarray()
    test_movie_target = test_movies.values()
    print len(test_movie_data), len(test_movie_target)

    gnb_fitted =  gnb.fit(train_movie_data, train_movie_target)

    print gnb_fitted.score(test_movie_data, test_movie_target)

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

    sampled_movies_per_decade = {decade: random.sample(movies, 1000) for (decade, movies) in movies_per_decade.iteritems()}
    train_movies_per_decade_sample = {decade: [movies[i] for i in range(len(movies)) if i % 2 == 0] for (decade, movies) in sampled_movies_per_decade.iteritems()}
    test_movies_per_decade_sample = {decade: [movies[i] for i in range(len(movies)) if i % 2 != 0] for (decade, movies) in sampled_movies_per_decade.iteritems()}

    sampled_movies =  reduce(lambda x, y: x+y, sampled_movies_per_decade.values())
    train_movies_sample =  reduce(lambda x, y: x+y, train_movies_per_decade_sample.values())
    test_movies_sample =  reduce(lambda x, y: x+y, test_movies_per_decade_sample.values())
    # plot_count_per_decade()
    # plot_count_per_decade_given_word("zombie", sampled_movies)

    # train_naive_bayes_model(train_movies_per_decade_sample)
    # true_positive = {'1930': 0, '1940': 0, '1950': 0, '1960': 0, '1970': 0, '1980': 0, '1990': 0, '2000': 0, '2010': 0}
    
    # for decade, movies in test_movies_per_decade_sample.iteritems():
    #     for movie in movies:
    #         log_posterior = test_naive_bayes(movie, [])
    #         if max(log_posterior.iteritems(), key=operator.itemgetter(1))[0] == decade:
    #             true_positive[decade] += 1
    # print true_positive

    # plot_posterior_histogram(test_movies_to_plot)

    # plot_cumulative_match_curve(train_movies_per_decade_sample, test_movies_per_decade_sample)
    # plot_confusion_matrix(train_movies_per_decade_sample, test_movies_per_decade_sample)

    # top_n_info_words(sampled_movies_per_decade, 10)

    # confusion_matrix_strip_top_100_informative_words(sampled_movies_per_decade, train_movies_per_decade_sample, test_movies_per_decade_sample)
    
    sklearn_naive_bayes(train_movies_sample, test_movies_sample)

    
    # len(all_movies)
    # => 379451
