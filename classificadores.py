import pandas as pd
from sklearn.calibration import cross_val_predict, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#region Validacao cruzada
def execute_model_multinomial(x_train, y_train):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(x_train, y_train)

    results_nb = cross_val_predict(nb_classifier, x_train, y_train, cv = 10)

    return results_nb

def execute_logistic_regression(x_train, y_train):
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42, multi_class='multinomial')
    lr_classifier.fit(x_train, y_train)

    results_lr = cross_val_predict(lr_classifier, x_train, y_train, cv = 10)

    return results_lr

def execute_SVC(x_train, y_train):
    svc_classifier = LinearSVC()
    svc_classifier.fit(x_train, y_train)

    results_svc = cross_val_predict(svc_classifier, x_train, y_train, cv = 10)

    return results_svc

def execute_random_forest(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)

    results_rf = cross_val_predict(rf_classifier, x_train, y_train, cv = 10)

    return results_rf
#endregion

def train(filePath, test_size):
    # Carregar o arquivo JSON em um DataFrame
    data = pd.read_json(filePath)

    # Dividir os dados em conjuntos de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(data['TweetContent'], data['IsRelated'], test_size=test_size, random_state=42)

    x_train = [' '.join(tweet) for tweet in x_train]
    x_test = [' '.join(tweet) for tweet in x_test]

    #CountVectorizer para frequencia
    vectorize = CountVectorizer()
    word_count_matrix = vectorize.fit_transform(x_train)

    count_list = word_count_matrix.toarray().sum(axis=0)
    word_list = vectorize.get_feature_names_out()

    word_freq = pd.DataFrame(count_list, index=word_list, columns=['Freq'])
    word_freq.sort_values(by='Freq', ascending=False).head(30)

    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(word_count_matrix)

    # Executando validação cruzada
    results_nb = execute_model_multinomial(x_train_tfidf, y_train)

    results_lr = execute_logistic_regression(x_train_tfidf, y_train)

    results_svc = execute_SVC(x_train_tfidf, y_train)

    results_rf = execute_random_forest(x_train_tfidf, y_train)

    print('results_nb')
    print(results_nb)
    print('results_lr')
    print(results_lr)
    print('results_svc')
    print(results_svc)
    print('results_rf')
    print(results_rf)