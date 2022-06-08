import nltk
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random
from nltk.metrics.distance  import edit_distance
from nltk.corpus import words
# nltk.download('reuters')
# nltk.download('punkt')
# nltk.download('words')






def sentenceBuilder(text,model,keepOld):

    if not keepOld: #check to create seperate sentances with the starting words or continue writing where the last finished
        text = text[:2]

    sentence_finished = False

    if type(list(model.keys())[1]) == type(tuple()):
        while not sentence_finished:
        # select a random probability threshold  
            r = random.random()
            accumulator = .0

            for word in model[tuple(text[-2:])].keys():
                accumulator += model[tuple(text[-2:])][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    text.append(word)
                    break

            if text[-2:] == [None, None]:
                sentence_finished = True
    else:
        while not sentence_finished:
        # select a random probability threshold  
            r = random.random()
            accumulator = .0

            for word in model[text[-1:][0]].keys():
                accumulator += model[text[-1:][0]][word]
                # select words that are above the probability threshold
                if accumulator >= r:
                    text.append(word)
                    break

            if text[-1:] == [None]:
                sentence_finished = True
    
    print (' '.join([t for t in text if t]))


def modelBuilder():

    # Create a placeholder for model
    model3 = defaultdict(lambda: defaultdict(lambda: 0))
    model2 = defaultdict(lambda: defaultdict(lambda: 0))

    corpus = preprocess(nltk.corpus.reuters.sents())


    # Count frequency of co-occurance  
    for sentence in corpus:
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model3[(w1, w2)][w3] += 1 #a tuple of the first two words is used as the key and the third
        for w1, w2 in bigrams(sentence, pad_right=True, pad_left=True):
            model2[w1][w2] += 1 #a tuple of the first two words is used as the key and the third
    
    # Let's transform the counts to probabilities

    model2 = modelProbabilities(model2)
    model3 = modelProbabilities(model3)
    print('Models Created')

    return model2,model3



def modelProbabilities(model):
    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count
    print('probability calculated')    
    return model


def preprocess(text):
    return text



# print(dict(model[text[0],text[1]]))
#sorted(dict(model["today","the"]),key=dict(model["today","the"]).get,reverse=True)[:5]


  



def spellcheck(correct_words,incorrect_word,return_max):
    #take list of correct words from dictionary OR model
    #take list of incorrect words flagged by the model
    #take maximum number of duggestions to return
    #return list of most similar words to the incorrect one based on edit distance

    distance = {}
    suggestions = []
    #correct_words = words.words()
    # list of incorrect spellings
    # that need to be corrected 
    #incorrect_word=['happpy', 'azmaing', 'intelliengt']
    
    # loop for finding correct spellings
    # based on edit distance and
    # printing the correct words
    for word in incorrect_word:
        for w in correct_words:
            if w[0]==word[0]:
                distance[w] = edit_distance(word, w)
        #print(sorted(distance, key = lambda val:val[0])[0][1]) #return top closest match words
        sorted_suggestions = sorted(distance.items(), key = lambda val:val[1])
        for i in range(return_max):
            suggestions.append(sorted_suggestions[i][0]) #append to list the top words with lowest edit distance

    return suggestions #list




def main():

    langModel2,langModel3 = modelBuilder() #create the language model based on Reuters corpus

    text = ["the", "president"]
    for i in range(5):
        sentenceBuilder(text,langModel3,True)


main()
#print(spellcheck(words.words(),['happpy'],5))