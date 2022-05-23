# ITC6010
Natural Language Processing

1. Overview
Your task in this assignment is to write a program that accepts as input a plain text document and extract some properties of the content.  Specifically, your program must output:
a.	Number of paragraphs.
b.	Number of sentences.
c.	Number of words (i.e., "tokens").
d.	Number of distinct words (i.e., "word types").
e.	List of word frequency counts.  Words are ordered by frequency (in the descending order), and words which have the same frequency count are ordered by lexicographical order (in the ascending order).
f.	Remove the stopwords (i.e., words that are frequent but do not contribute much to the meaning of a sentence.)  A list of stopwords is provided for English.
g.	How would what you did be different if you did it for another language (e.g. Greek or French). List as many changes in your approach as you can. Do you think there can be a universal methodology for identifying and counting words / sentences / paragraphs, or is it language specific?
The purpose of this assignment is that you write the code from scratch, without using any library function (except for those that are fundamentally necessary such as those for file I/O or collection classes).
Exception: You can use library in sentence segmentation.  In general, a document can be formatted in a variety of ways, especially regarding line breaks.  To save time, you are allowed to use a library function (though only for sentence segmentation).
•	In Python, you may use sent_tokenize() in Natural Language Toolkit (NLTK) toolkit.  But do not use other NLTK functions (such as word_tokenize()) or external libraries/packages.
Try to implement all parts by yourselves and experiment if you have time on how to make your code efficient: e.g., in memory, or in processing time. For example, read the text just one time or write fast code by simplifying some operations.
