import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import emoji

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Função para remover emojis
def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)

# Função para remover URLs, hashtags, menções e emoticons
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    text = remove_emojis(text)
    text = re.sub(r':\)|:-\)|:\(|:-\(', '', text)

    return text

def tokenize_file(data):
    tokenized_documents = []
    for document in data:
        content = document['TweetContent']
        
        words = word_tokenize(content)
        
        processed_words = []
        for word in words:
            processed_words.extend([word])
        
        document['TweetContent'] = processed_words
        
        tokenized_documents.append(document)
    
    return tokenized_documents

def preprocess_documents(data):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    preprocessed_documents = []
    for document in data:
        content = document['TweetContent']
        
        content = content.lower()
        content = preprocess_text(content)
        words = word_tokenize(content)
        
        processed_words = []
        for word in words:
            if word.isalpha() and word not in stop_words:
                lemma = lemmatizer.lemmatize(word)
                processed_words.extend([lemma])
        
        document['TweetContent'] = processed_words
        
        preprocessed_documents.append(document)
    
    return preprocessed_documents
