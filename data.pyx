from nltk.corpus import wordnet
from nltk.corpus import words
import numpy as np
from random import randrange
from math import exp, pi

words_data = list(wordnet.words()) + list(words.words())
words_data = list(set(words_data))[:1000]
size = len(words_data)
X = []
Y = []
max_length = len(max(words_data,key=len))

def shuffle(a, b, seed):
    x, y = a, b
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(x)
    rand_state.seed(seed)
    rand_state.shuffle(y)
    return x, y

def encode_word(word):
    sumB, change = 0.0, 0.0                                     #sum of binaries and their avg. change
    Y = []                                                      #container for integer of characters
    size = len(word)
    for i in range(size):
        binary = int(format(ord(word[i]), 'b'), 2)              #convert each char to integer via binary
        sumB += exp(-binary) + i                                
        Y.append(binary)
        if (i != 0):
            change += Y[-1] - binary

    Y = np.array(Y)
    Y_out = np.zeros((1, max_length))
    Y_out[0, :Y.shape[0]] = Y

    return np.matrix([size, sumB * size, change / size]), np.matrix(Y_out)

print("Encoding Words...")
for word in words_data:
    x, y = encode_word(word)
    X.append(x)
    Y.append(y)
print("Done...!!")

X, Y = shuffle(X, Y, randrange(1, len(X)))
#X = np.array(X)
#Y = np.array(Y)

