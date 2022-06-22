import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#import natural language tool kittens and the stemmer, meow!

import numpy
import tflearn
import tensorflow
import random

#import some other kittens (these are packages, see notes for what they do)

import json

with open('intents.json') as file:
    data = json.load(file)

#import json file with the intents (this file is a dictionairy/object that contains dictionaries/objects for every 'tag'=intent, which contain nested lists of 'patterns'=training phrases and 'responses')

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

#preprocessing of the data : tokenize, create lists for words and tags (of the existing intents) and add new tags (for new intents)

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

#create a list of all the unique stemmed words (turn all the words into lower case, use a set for getting the unique values and get the stem of all the words). Questionmarks are not words. avo: add esclamation marks/... to these exceptions? Yes.
#coppi: sorting the list why? To match with the lables list (as this is sorted too)? Wasn't a bag of words unsorted?
#avo: the order of the words in the sentence is lost, but the lists 'words' and 'labels' are sorted. RoboCoppi only knows the presence of words in our models vocabulary, not their order in the sentence.
#techtim: "I just sort these because it looks a bit nicer" (avo: so I guess it is not necessary and same thing for 'labels').

training = []
output = []

#create empty lists for formatting our input and output.

out_empty = [0 for _ in range(len(labels))]

#techtim: every class (tag in this case) in our output is 0 to start with
#avo: the output lists are the length of how many tags (classes) there are. Looping through those lists the value is set to 0 for each tag/class.

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

#we could have stemmed the words when preprocessing the data (block starting at line 26, the preprocessing block).

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

#represent each sentence with the length of the amount of words in our models vocabulary.
#each position in the list will represent a word from our vocabulary.
#if the position in the list is a 1, this means the word exists in our sentence. 0 for not present.
#creates a new binary feature for each possible category and assigns a value of 1 to the feature of each sample that corresponds to its original category. This is called One Hot Encoding, where hot stands for True. The truth is hot indeed.

training = numpy.array(training)
output = numpy.array(output)

#convert our training data and output to numpy arrays.

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#define the architecture of our model. This model is a feed-forward neural network with two hidden layers. You can change the number of neurons (8 per layer in this example) in the layers to see what happens.

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

#training and saving our model.
#fit our data to the model. The number of epochs we set is the amount of times that the model will see the same information while training. You can change its value to see what happens.
#save our data to the file model.tflearn for use in other scripts.

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

        if results[result_index] > 0.7:
#This sets the treshold to 70% correctness
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I totally hear you. I just can't really wrap my whiskers around it yet.")

chat()

# Get some input from the user
# It doesn't seem to get beyond this step (script works up to line 101). I went back to the basics and changed to vs code. After getting really confused and trying the following: Installed 'Random Everything' (in Sublime Text) and 'Anaconda', as PIP connecting to 1 Python only situation (1 error message to solve left), no change.
# Convert it to a bag of words
# Get a prediction from the model
# Find the most probable class
# Pick a response from that class
