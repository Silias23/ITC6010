import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')


'''
DOWNLOAD DATA FROM https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
AND FIX THE read_csv() line to correct directory
'''

def merge_columns(df):
    #used to merge the title and the text to one single column for processing

    print(f'N/A values per column: \n{df.isna().sum()}')

    df[df.isna().any(axis=1)] #check the rows that have NA values

    df.dropna(inplace=True)#drop na values and reset the indexes
    df.reset_index(drop=True, inplace=True)

    df['Full Text'] = df['Title'] + '. ' + df['Review'] #merge the title with the body of the review in one column

    df.drop(['Title','Review'], axis=1, inplace=True) #drop the columns that will not be used

    return df


def preprocess(df):

    df['Polarity'] = df['Polarity'].replace([1,2],['0','1']) #fix 1,2 to corespond to 0,1 negative/positive
    df['Full Text'] = df['Full Text'].str.lower() #turn all text to lowercase
    df['Full Text'] = df['Full Text'].str.replace(r"@"," at ") #change emails and tags to not use the symbol
    df['Full Text'] = df['Full Text'].str.replace("#\S+ "," ") #remove all hashtags from the text \S is used to include all hashtags with or without symbols 

    '''
    TODO: add a way to remove all symbols and numbers from a sentence without leaving random characters behind. 
    Need to match the entire sequence of characters and remove it. 
    exception is fullstops '.' we need to keep them to signify end of sentences. 

    Example:
    Sigma 17-35mm f/2.8-4 EX DG IF HSM Aspherical --> is the preprocessed sentence. after applying standard regex replace to replace the numbers and symbols we are left with

    Sigma mm f EX DG IF HSM Aspherical --> need to remove 'mm', 'f', and any other words that dont make sense

    one idea is to split each sentence and remove all words that do not appear in the english dictionary << VERY costly

    TODO: add any other cleaning and preprocessing you want

    '''
    return df



def text_lemmatize(data): #broken takes each individual character need to take words
    lemmatizer = WordNetLemmatizer()
    out_data=""
    data = data.split(' ')
    for words in data:
        out_data+= lemmatizer.lemmatize(words)

    return out_data


headers = ['Polarity', 'Title', 'Review']

data = pd.read_csv('FinalProject/test.csv', names=headers)
data = data[:1000] #for now keep 1000 rows to make it run faster//will remove later

data = merge_columns(data)
data = preprocess(data)
#data["Full Text"].apply(lambda x: text_lemmatize(x))

print(data)



