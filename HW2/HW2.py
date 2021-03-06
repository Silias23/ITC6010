#"""
# Run Instructions:
# run HW2.py
# Terminal prompt will ask for setup variables
    # number of sentences: select number of sentences to be generated by the system
    # error: specifies the number of errors in each sentence. this is not absolute, it gets modified by randomizer so max number of errors is input number. by experimenting 20 is a good weight for randomizing words
    # sugg: number of suggestions to output for each error found. the list first gets populated by the model's suggestions and any open spaces are filled by generic unigram model uding edit distance
    # auto: enable automatic spellchecker. will always use the word with smallest edit distance as suggested word --- this has some issues, in large generated sentences some words will throw the spellchecker in a loop trying to guess the same word
    # new: if multiple sentences are generated, specify if each sentence will start using the original bigram or each sentence will have a new begining word
# 
# A sentence will be generated automatically using the model created from Reuters corpus -- that is a sentence generated real time, it does not exist in the corpus
# The sentence will be altered and random spelling mistakes will be added to each word
# the user will be presented with the altered sentence 
# for each word that is flagged as wrong by the system, the user will have the option to choose one of the words/suggestions on screen. 
# After the user chooses the word they want to replace the wrong word with, the program continues until all words in the sentence are fixed  
#"""


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
    #used to create the random sentences based off the model
    #output is sentence

    if not keepOld: #check to create seperate sentances with the starting words or continue writing where the last finished
        text = text[:2]
    if keepOld:
        text = text[-2:] #if old sentence is kept, then keep the text but keep only the last two words as it is a new sentence (sentences end with None None)

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
    elif type(list(model.keys())[1]) == type(str()):#used if bigram model is selected
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
    
    return text[2:] 


def modelBuilder(corp):
    #main model function
    #texts as input
    #preprocess and build model based on bigrams and trigrams
    #output two models
    print("Building Model")

    corpus = preprocess(corp)
    model2,model3 = frequencies(corpus)
    
    # Let's transform the counts to probabilities
    model2 = modelProbabilities(model2)
    model3 = modelProbabilities(model3)
    print('Models Created')
    
    return model2,model3


def frequencies(sentences):
    # Create a placeholder for model
    trigram = defaultdict(lambda: defaultdict(lambda: 0))
    bigram = defaultdict(lambda: defaultdict(lambda: 0))

    # Count frequency of co-occurance  
    for sentence in sentences:
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            trigram[(w1, w2)][w3] += 1 #a tuple of the first two words is used as the key and the third
        for w1, w2 in bigrams(sentence, pad_right=True, pad_left=True):
            bigram[w1][w2] += 1 #a single word is used as the key for the bigrams
    
    # import pandas as pd
    # df = pd.DataFrame(columns=('bigram','frequency'))
    # for key in bigram.keys():
    #     c = 0
    #     for item in bigram[key].keys():
    #         c += bigram[key][item]
    #     new_row = {'bigram':key,'frequency':c} 
    #     df=df.append(new_row, ignore_index=True)   
        
    return bigram,trigram


def modelProbabilities(model):
    #calculate the probablities for each ngram to exist 
    #this can be changed to include LaPlace probabilities model
    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count  
    return model


def preprocess(text):
    #change all sentences to lowercase
    b=[]
    for sentence in text:
        a = (map(lambda x: x.lower(), sentence))
        b.append(list(a))
    return b


def sentenceMixer(sentence,weight):
    #randomize the words in each sentence
    #reverse edit distance to mix the characters in each word
    replacement_chars='abcdefghijklmnopqrstuvwxyz'

    for i in range(int(weight*10*random.random())): #number of changes to occur random used to change errors in sentence dynamically, max being weight*1
        r1 = random.randrange(2,len(sentence)-2) #random word to change. cannot be the first two because no trigram match, cannot be the last two because sentence ends in double NoneTypes
        if len(sentence[r1])==1: #if word is single letter no point in changing it, may be a symbol and will throw off the model
            continue
        elif sentence[r1].isdigit() or any(not c.isalnum() for c in sentence[r1]): #skip the words that contain numbers and symbols
            continue
        else:
            r2 = random.randrange(1,len(sentence[r1])) # select random position in word
            r3 = random.randint(0,2)
            if r3 == 0:
                char = replacement_chars[random.randint(0,len(replacement_chars)-1)]
                sentence[r1] = sentence[r1][:r2] + char + sentence[r1][r2+1:]#replace character at index with a random letter from the alphabet
            elif r3 == 1:
                if len(sentence[r1]) > 3:
                    sentence[r1] = sentence[r1][:r2] + sentence[r1][r2+1:] #remove character at random index
            elif r3 == 2:
                char = replacement_chars[random.randint(0,len(replacement_chars)-1)] #add character at random index
                sentence[r1] = sentence[r1][:r2] + char + sentence[r1][r2:]

    return sentence


def spellcheck(correct_words,incorrect_word,return_max):
    #take list of correct words from model and dictionary
    #take list of incorrect words flagged by the model
    #take maximum number of duggestions to return
    #return list of most similar words to the incorrect one based on edit distance


    distance = {}
    extra_distance={}
    suggestions = []
    
    #find edit distance for each incorrect word using the suggested words from the language model
    for word in incorrect_word:
        for w in correct_words:
            try:
                if w[0]==word[0]:
                    distance[w] = edit_distance(word, w)
            except:
                continue
        #if the model did not cover the suggestion limit, fill the rest of the spots with suggestions from the words() dictionary
        if len(distance) < return_max: 
            for w in words.words():
                try:
                    if w[0]==word[0]:
                        extra_distance[w] = edit_distance(word, w)
                except:
                    continue
            sorted_distance = sorted(extra_distance.items(), key = lambda val:val[1])
            for i in range(return_max - len(distance)):
                distance[sorted_distance[i][0]]=sorted_distance[i][1]    #if word exists in both models then the suggestion list will be one short. suggestions get MERGED not ADDED

        sorted_suggestions = sorted(distance.items(), key = lambda val:val[1])
    for i in range(min(return_max,len(sorted_suggestions))): #min() used for error handling in case words() model does not return adequate results
        suggestions.append(sorted_suggestions[i][0]) #append to list the top words with lowest edit distance
        

    return suggestions #list


def mistakeFinder(sentence,bigram,trigram,max_sugg,auto):
    #takes a sentence and splits in in trigrams
    #if the trigram does not exist (last word in trigram is wrong)
    #run spellchecker to return suggestions and user gives input
    #wrong word gets replaced by user selection

    sentence = list(filter(None,sentence))

    i=0
    for w1, w2, w3 in trigrams(sentence, pad_right=True):
        i += 1
        if w3 not in list(trigram[(w1,w2)].keys()):
            correctWord = spellcheck(list(trigram[(w1,w2)].keys()),[w3],max_sugg)
            if auto: #auto always uses the first siggested word -- the one with smallest edit distance
                c = 1
            else:
                c = userprompt(w3,correctWord)
            sentence = sentence[:i+1]+[correctWord[c-1]]+sentence[i+2:]
            print(f'Wrong word:{w3}, Corrected:{correctWord[c-1]}')
            return sentence,False
        else:
            continue
    return sentence,True

                
def userprompt(w3,correctWord):
    #for each incorrect word the user is prompted to select a replacement
    #returns the index of the word selected
    print(f'Incorrect word found: {w3} \n please type the number corresponding to the correct word:')

    while True:
        for j in range(len(correctWord)):
            print(f'{j+1}. {correctWord[j]}')  #print all suggestions

        try:      
            c = int(input())
        except:
            print("Please select a number")
            continue
        if c>len(correctWord) and c < 0 :
            print("invalid number. Please select one from the options:")
            continue
        else:
            break
    return c


def setup():
#take user input for initializing the models and spellchecker
    num = 1
    error = 20
    sugg = 5
    auto = 'f'
    new = True

    while True:
        try:
            print("please specify the number of sentences to be generated (Default=1)(Max=5)")
            num = int(input())
        except:
            print("Please insert a number")
        if num >= 1 and num <= 5:
            break
        else:
            print("Number must be between 1 and 5")
            continue
    if num >1:
        while True:
            try:
                print(f"Do you want to create {num} sentences using the same starting words? (the company ...)\nIf No then the subsequent sentences will be generated at random\n(Y)es (N)o (Default:Yes)")
                new = input()
            except:
                print("Please type (Y)es (N)o")
            if new in ['y','Y','Yes','1']:
                new = True    
                break
            elif new in ['n','N','No','0']:
                new = False
                break
            else:
                print("Please type (Y)es (N)o")
                continue
    while True:
        try:
            print("please specify the weight of mistakes for each sentence \n(e.g. \'2\' sets the maximum number of insertions/deletions/substitutions to 20)\nDefault:2, Max:5")
            error = int(input())
        except:
            print("Please insert a number")
        if error >= 1 and error <= 5:
            break
        else:
            print("Number must be between 1 and 5")
            continue
    while True:
        try:
            print("please specify the number of suggested words to appear for each mistake\nDefault:5 Max:10")
            sugg = int(input())
        except:
            print("Please insert a number")
        if sugg >= 1 and sugg <= 10:
            break
        else:
            print("Number must be between 1 and 10")
            continue
    while True:
        try:
            print('Do you want to enable autorun? (experimental)\n(the program will automatically choose the first suggestion for each word with a mistake)\nType (Y)es or (N)o')
            auto = input()
        except:
            print("Please type (Y)es (N)o")
        if auto in ['n','N','No','0']:
            auto = False
            break
        elif auto in ['y','Y','Yes','1']:
            auto = True
            break
        else:
            print("Please type (Y)es (N)o")
            continue

    print('Setup Complete')
    return num,error,sugg,auto,new

def main():
    numSentence,errorWeight,maxsuggestions,automode,StartNew = setup()
    langModel2,langModel3 = modelBuilder(nltk.corpus.reuters.sents()) #create the language model based on Reuters corpus
    newSentences = [["today", "the"]]
    mixedSentences = []
    for i in range(numSentence):
        newSentences.append(sentenceBuilder(newSentences[i],langModel3,StartNew))
    newSentences[0 : 2] = [newSentences[0]+newSentences[1]]
    print('Sentences Created')
    print('---------------------------------')
    for i in range(len(newSentences)):
        print (' '.join([t for t in newSentences[i] if t]))
        mixedSentences.append(sentenceMixer(newSentences[i],errorWeight))
        done = False
        while not done:
            print (' '.join([t for t in mixedSentences[i] if t]))
            mixedSentences[i],done = mistakeFinder(mixedSentences[i],langModel2,langModel3,maxsuggestions,automode)
        print('Sentence corrected')
        print('---------------------------------')


    #print(spellcheck(words.words(),['hsppt'],5))



if __name__ == '__main__':
    main()
