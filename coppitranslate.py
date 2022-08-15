# Import the natural language tool kit and the stemmer

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Import the following packages

import numpy
import tflearn
import tensorflow
import random

# Import json file with the training data (the intents)

import json

with open('intents.json') as file:
    data = json.load(file)
    
# Preprocessing of the data : create lists for the words in every training_phrase, for the intent_names, for the tokenized_training_phrases and for the intent_name_tags.
words = []
intent_names = []
tokenized_training_phrases = []
intent_name_tags = []

# Looping through the intents.json file: tokenize the training_phrases and add them to the tokenized_training_phrases list. And add intent_name for every training_phrase to the intent_name_tags list.

for intent in data['intents']:
    for training_phrase in intent['training_phrases']:
        preprocessed_words = nltk.word_tokenize(training_phrase)
        words.extend(preprocessed_words)
        tokenized_training_phrases.append(preprocessed_words)
        intent_name_tags.append(intent['intent_name_tag'])
        
    if intent['intent_name_tag'] not in intent_names:
        intent_names.append(intent['intent_name_tag'])

# Create a list of all the unique stemmed words (turn all the words into lower case, use a set for getting the unique values and get the stem of all the words). Questionmarks are not words. //TODO add esclamation marks/... to these exceptions.
# The order of the words in the sentence is lost, but the lists 'words' and 'intent_names' are sorted. Robocoppi only knows the presence of words in his model's vocabulary, not their order in the sentence.

words = [stemmer.stem(word.lower()) for word in words if word != "?"]
words = sorted(list(set(words)))

intent_names = sorted(intent_names)

# Create empty lists for formatting our input and output.

training = []
output = []

# The output lists are the length of how many intent names there are. Looping through those lists the value is set to 0 for each intent_name.
# Hence every intent_name in our output has a value of 0 to start with.

out_empty = [0 for _ in range(len(intent_names))]

for i, token in enumerate(tokenized_training_phrases):
    bag = []

    print(i)
    print(token)
    print(tokenized_training_phrases)
    print(intent_name_tags)
    