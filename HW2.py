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
    if keepOld:
        text = text[-2:]


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
    elif type(list(model.keys())[1]) == type(str()):
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
    return text[2:]




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
    #TODO change everything to lowercase, maybe remove punctuation marks?
    return text



# print(dict(model[text[0],text[1]]))
#sorted(dict(model["today","the"]),key=dict(model["today","the"]).get,reverse=True)[:5]


def sentenceMixer(sentence,weight):
    replacement_chars='abcdefghijklmnopqrstuvwxyz'
    for i in range(int(weight*random.random())): #number of changes to occur
        r1 = random.randrange(2,len(sentence)-2) #random word to change. cannot be the first two because no trigram match, cannot be the last two because sentence ends in double NoneTypes
        if len(sentence[r1])==1: #if word is single letter no point in changing it, may be a symbol and will throw off the model
            continue
        else:
            r2 = random.randrange(1,len(sentence[r1])) # select random letter in word
        char = replacement_chars[random.randint(0,len(replacement_chars)-1)]
        sentence[r1] = sentence[r1].replace(sentence[r1][r2],char) #replace all instances of the letter with a random letter from the alphabet
    print(sentence)    

    return sentence
  



def spellcheck(correct_words,incorrect_word,return_max):
    #take list of correct words from dictionary OR model
    #take list of incorrect words flagged by the model
    #take maximum number of duggestions to return
    #return list of most similar words to the incorrect one based on edit distance

    distance = {}
    suggestions = []
    
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

    newSentences = [["today", "the"]]
    mixedSentences = []
    for i in range(5):
        newSentences.append(sentenceBuilder(newSentences[i],langModel3,True))
    newSentences[0 : 2] = [newSentences[0]+newSentences[1]]
    for i in range(len(newSentences)):
        mixedSentences.append(sentenceMixer(newSentences[i],10))

    #TODO create def to find mistakes in sentence
    # add a df with suggestions based on words() and the two models
    # OR list all suggestions in prompt for user to select correct word and move forward to next mistake
    # ^ if this then add setup prompt to create max mistakes per sentence, max sentence to be created etc
    #TODO maybe handle 1st and 2nd words?
    #TODO after finding mistake in word load word and model to spelchecker, DIDNT WORK index out of range need to fix

    print(spellcheck(words.words(),['hsppt'],5))



if __name__ == '__main__':
    main()
