import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model

def execute_naive_bayes(filePath, test_size):
    # Carregar o arquivo JSON em um DataFrame
    data = pd.read_json(filePath)

    # Dividir os dados em conjuntos de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(data['TweetContent'], data['IsRelated'], test_size=test_size, random_state=42)

    x_train = [' '.join(tweet) for tweet in x_train]
    x_test = [' '.join(tweet) for tweet in x_test]
    # y_train = [str(valueIsRelated) for valueIsRelated in y_train]
    # y_test = [str(valueIsRelated) for valueIsRelated in y_test]

    # print('x_train')
    # print(x_train)

    return

    # Criar uma matriz TF-IDF a partir dos tweets de treinamento
    tfidf_vectorizer = TfidfVectorizer()
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

    print('x_train_tfidf')
    print(x_train_tfidf.toarray())

    # Inicializar e treinar o modelo Naive Bayes
    # naive_bayes_classifier = MultinomialNB()
    # naive_bayes_classifier.fit(x_train_tfidf, y_train)

    # Transformar os tweets de teste em vetores TF-IDF
    # x_test_tfidf = tfidf_vectorizer.transform(x_test)

    execute_model_multinomial(x_train_tfidf, y_train)

    # # Fazer previsões
    # predictions = naive_bayes_classifier.predict(x_test_tfidf)

    # # Avaliar o desempenho do modelo
    # accuracy = accuracy_score(y_test, predictions)
    # print(f'Acurácia do modelo Naive Bayes: {accuracy}')

def execute_model_multinomial(x_train, y_train):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(x_train, y_train)

    results_nb = cross_val_predict(nb_classifier, x_train, y_train, cv = 10)

    return nb_classifier, results_nb