#import natural language tool kit and the stemmer

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# FIXME - move comments so they precede the relevant code

# import the following packages //FIXME - resolve issues with these packages

import numpy
import tflearn
import tensorflow
import random

# import json file with the intents

import json

with open('intents.json') as file:
    data = json.load(file)
    
#preprocessing of the data : tokenize, create lists for words and add tags to the intents

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

#create a list of all the unique stemmed words (turn all the words into lower case, use a set for getting the unique values and get the stem of all the words). Questionmarks are not words. //TODO add esclamation marks/... to these exceptions.
#the order of the words in the sentence is lost, but the lists 'words' and 'labels' are sorted. RoboCoppi only knows the presence of words in our models vocabulary, not their order in the sentence.

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

#create empty lists for formatting our input and output.

training = []
output = []

#every class (tag in this case) in our output is 0 to start with
#the output lists are the length of how many tags (classes) there are. Looping through those lists the value is set to 0 for each tag/class.

out_empty = [0 for _ in range(len(labels))]

#we could have stemmed the words when preprocessing the data (block starting at line 26, the preprocessing block).

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]
    
#represent each sentence with the length of the amount of words in our models vocabulary.
#each position in the list will represent a word from our vocabulary.
#if the position in the list is a 1, this means the word exists in our sentence. 0 for not present.
#creates a new binary feature for each possible category and assigns a value of 1 to the feature of each sample that corresponds to its original category. This is called One Hot Encoding, where hot stands for True.
   
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)
    
#convert our training data and output to numpy arrays.

training = numpy.array(training)
output = numpy.array(output)

#define the architecture of our model. This model is a feed-forward neural network with two hidden layers. You can change the number of neurons (8 per layer in this example) in the layers to see what happens.

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#training and saving our model.
#fit our data to the model. The number of epochs we set is the amount of times that the model will see the same information while training. You can change its value to see what happens.
#save our data to the file model.tflearn for use in other scripts.

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

# TODO - review following code

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
#This sets the treshold to 70% correctness

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I totally hear you. I just can't really wrap my whiskers around it yet.")

chat()

# Get some input from the user
# Convert it to a bag of words
# Get a prediction from the model
# Find the most probable class
# Pick a response from that class
