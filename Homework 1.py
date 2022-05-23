# a.	Number of paragraphs.
# b.	Number of sentences.
# c.	Number of words (i.e., "tokens").
# d.	Number of distinct words (i.e., "word types").
# e.	List of word frequency counts.  Words are ordered by frequency (in the descending order), and words which have the same frequency count are ordered by lexicographical order (in the ascending order).
# f.	Remove the stopwords (i.e., words that are frequent but do not contribute much to the meaning of a sentence.)  A list of stopwords is provided for English.
# g.	How would what you did be different if you did it for another language (e.g. Greek or French). List as many changes in your approach as you can. Do you think there can be a universal methodology for identifying and counting words / sentences / paragraphs, or is it language specific?
import nltk
#nltk.download('punkt')
f = open('ITC6010 NLP\Sample Text.txt', 'r')
g = open('ITC6010 NLP\English Stopwords.txt', 'r')
stopWords = g.read().splitlines()
paragraphs = f.readlines()
sentences = []
SumSentences = 0
wordlist = []
frequency = {}

#for each paragraph split into sentences and for each sentence split into words
for i,paragraph in enumerate(paragraphs):
    sentences.append(nltk.sent_tokenize(paragraph))
    SumSentences += len(sentences[i])
    for sentence in sentences[i]:
        sentence = sentence[:-1] #remove the punctuation from the end of each sentence
        words = sentence.split(" ")
        wordlist.extend(words) #add all the words of the sentence to a new array
        for word in words:
            if word in frequency:
                frequency[word] += 1
            else:
                frequency[word] = 1

frequency_NoStopWords = frequency.copy()

for word in stopWords:
    if word in frequency_NoStopWords:
        del frequency_NoStopWords[word]






print(f'Number of paragraphs: {len(paragraphs)}, Number of Sentences: {SumSentences}, Number of words: {len(wordlist)}, Unique words: {len(frequency)}, Unique words without stop words: {len(frequency_NoStopWords)}')


for k,v in sorted(frequency.items(),key=lambda p:p[1],reverse=True):
    print(k, v)