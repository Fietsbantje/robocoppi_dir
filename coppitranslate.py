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
    
# Preprocessing of the data. Create lists for: the words in every training_phrase, for the intent_names, for the tokenized_training_phrases and for the (matching) intent_name_tags.
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
# Represent each sentence with the length of the amount of words in our models vocabulary.
# Each position in the list will represent a word from our vocabulary.

out_empty = [0 for _ in range(len(intent_names))]

# Create a bag of words for training the model.

for i, token in enumerate(tokenized_training_phrases):
    bag = []

    preprocessed_words = [stemmer.stem(word.lower()) for word in token]
    
# If the position in the list is a 1, this means the word exists in our sentence. 0 for not present.

    for word in words:
        if word in preprocessed_words:
            bag.append(1)
        else:
            bag.append(0)

# Create a new binary feature for each possible category and assign a value of 1 to the feature of each sample that corresponds to its original category. This is called "one hot encoding", where "hot" stands for True.
   
    output_row = out_empty[:]
    output_row[intent_names.index(intent_name_tags[i])] = 1

    training.append(bag)
    output.append(output_row)
    
# Convert our training data and output to numpy arrays.

training = numpy.array(training)
output = numpy.array(output)

# Define the architecture of our model. This model is a feed-forward neural network with two hidden layers. You can change the number of neurons (8 per layer in this example) in the layers to see what happens.

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Training and saving our model.
# Fit our data to the model. The number of epochs we set is the amount of times that the model will see the same information while training. You can change its value to see what happens.
# Save our data to the file model.tflearn for use in other scripts.

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

# Convert input into a bag of words.

def bag_of_words(input_message, words):
    bag = [0 for _ in range(len(words))]

    tokenized_input_message = nltk.word_tokenize(input_message)
    stemmed_input_message = [stemmer.stem(word.lower()) for word in tokenized_input_message]

    for input_word in stemmed_input_message:
        for i, word in enumerate(words):
            if word == input_word:
                bag[i] = 1
            
    return numpy.array(bag)

# Get input from the user.
# If input is 'basta' stop the program.

def chat():
    print("Talk to me whenever you are ready! Type basta to stop")
    while True:
        input_message = input("You: ")
        if input_message.lower() == "basta":
            break

# Get a prediction from the model.

        results = model.predict([bag_of_words(input_message, words)])[0]
        results_index = numpy.argmax(results)
        prediction = intent_names[results_index]
        
# Set the treshold to 70% correctness.
# Find the most probable intent.
# Pick a response from that intent randomly.


        if results[results_index] > 0.7:
            for tag in data["intents"]:
                if tag['intent_name_tag'] == prediction:
                    responses = tag['responses']

            print(random.choice(responses))
        else:
            print("I totally hear you. I just can't really wrap my whiskers around it yet.")

chat()
    