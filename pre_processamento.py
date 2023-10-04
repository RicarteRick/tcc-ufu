import re
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from util_files import openFile, saveJsonFile
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

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

def get_word_freq(data):
    # CountVectorizer para frequencia
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(data)

    count_list = word_count_matrix.toarray().sum(axis=0)
    word_list = vectorizer.get_feature_names_out()

    word_freq = pd.DataFrame(count_list, index=word_list, columns=['Freq'])

    return word_freq

def generate_pareto_chart(docs_path, file_raw, file_preprocessed):
    #region arquivo não pré-processado
    data_raw = pd.read_json(docs_path+file_raw)
    data_raw_text = data_raw['TweetContent'].apply(lambda tweet: ' '.join(tweet))

    word_freq_raw = get_word_freq(data_raw_text)
    top_terms_raw = word_freq_raw.sort_values(by='Freq', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(top_terms_raw.index, top_terms_raw['Freq'], alpha=0.7, color='b', width=0.6, label='Frequência do termo')
    plt.xlabel('Termos')
    plt.ylabel('Frequência')
    plt.xticks(rotation=90)
    plt.legend(loc='upper right')
    plt.savefig(docs_path+'pareto_chart_raw.png', bbox_inches='tight', dpi=300)
    #endregion

    #region arquivo pré-processado
    data_preprocessed = pd.read_json(docs_path+file_preprocessed)
    data_preprocessed_text = data_preprocessed['TweetContent'].apply(lambda tweet: ' '.join(tweet))
    
    word_freq_preprocessed = get_word_freq(data_preprocessed_text)
    top_terms_preprocessed = word_freq_preprocessed.sort_values(by='Freq', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.bar(top_terms_preprocessed.index, top_terms_preprocessed['Freq'], alpha=0.7, color='b', width=0.6, label='Frequência do termo')
    plt.xlabel('Termos')
    plt.ylabel('Frequência')
    plt.xticks(rotation=90)
    plt.legend(loc='upper right')
    plt.savefig(docs_path+'pareto_chart_preprocessed.png', bbox_inches='tight', dpi=300)
    #endregion

def generate_word_cloud(docs_path, file_raw, file_preprocessed):
    #region arquivo não pré-processado
    data_raw = pd.read_json(docs_path+file_raw)
    data_raw_text = data_raw['TweetContent'].apply(lambda tweet: ' '.join(tweet))
    combined_text_raw = ' '.join(data_raw_text)

    wordcloud_raw = WordCloud(width=800, height=500, background_color='white').generate(combined_text_raw)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_raw, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(docs_path+'word_cloud_raw.png', bbox_inches='tight', dpi=300)
    #endregion

    #region arquivo pré-processado
    data_preprocessed = pd.read_json(docs_path+file_preprocessed)
    data_preprocessed_text = data_preprocessed['TweetContent'].apply(lambda tweet: ' '.join(tweet))
    combined_text_preprocessed = ' '.join(data_preprocessed_text)

    wordcloud_preprocessed = WordCloud(width=800, height=500, background_color='white').generate(combined_text_preprocessed)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_preprocessed, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(docs_path+'word_cloud_preprocessed.png', bbox_inches='tight', dpi=300)
    #endregion

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

def preprocess(docsPath, rawFileName, preprocessedFileName, tokenizedRawFileName):
    # carregando arquivo
    rawData = openFile(docsPath+rawFileName)

    # pré-processando
    preprocessedData = preprocess_documents(rawData)

    # salvando novo arquivo pré-processado
    saveJsonFile(docsPath+preprocessedFileName, preprocessedData)

    # salvando arquivo sem pré-processamento, mas tokenizado
    rawData2 = openFile(docsPath+rawFileName)
    tokenizedRawData = tokenize_file(rawData2)
    saveJsonFile(docsPath+tokenizedRawFileName, tokenizedRawData)

    print('Documentos salvos')