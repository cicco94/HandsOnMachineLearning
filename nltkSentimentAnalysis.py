# !pip install nltk
import nltk
#nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

from featx import bag_of_words
from featx import label_feats_from_corpus 
from featx import split_label_feats 

lfeats = label_feats_from_corpus(movie_reviews)

train_feats, test_feats = split_label_feats(lfeats, split=0.75)

from nltk.classify import NaiveBayesClassifier

nb_classifier = NaiveBayesClassifier.train(train_feats)
print(nb_classifier.labels())

review1 = bag_of_words(['the', 'plot', 'was', 'ludicrous'])
print(nb_classifier.classify(review1))

review2 = bag_of_words(['kate', 'winslet', 'is', 'accessible'])
print(nb_classifier.classify(review2))

from nltk.classify.util import accuracy

print(accuracy(nb_classifier, test_feats))