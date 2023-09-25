import pandas as pd
from sklearn.calibration import cross_val_predict, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#region Validacao cruzada
def execute_naive_bayes(x_train, y_train, k_folds):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(x_train, y_train)

    results_nb = cross_val_predict(nb_classifier, x_train, y_train, cv = k_folds)

    return results_nb

def execute_logistic_regression(x_train, y_train, k_folds):
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42, multi_class='multinomial')
    lr_classifier.fit(x_train, y_train)

    results_lr = cross_val_predict(lr_classifier, x_train, y_train, cv = k_folds)

    return results_lr

def execute_SVM(x_train, y_train, k_folds):
    svm_classifier = LinearSVC()
    svm_classifier.fit(x_train, y_train)

    results_svm = cross_val_predict(svm_classifier, x_train, y_train, cv = k_folds)

    return results_svm

def execute_random_forest(x_train, y_train, k_folds):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)

    results_rf = cross_val_predict(rf_classifier, x_train, y_train, cv = k_folds)

    return results_rf
#endregion

def train(filePath, k_folds):
    # Carregar o arquivo JSON em um DataFrame
    data = pd.read_json(filePath)

    # Setando os conjuntos de dados e classes
    x_train = data['TweetContent']
    y_train = data['IsRelated']

    x_train = [' '.join(tweet) for tweet in x_train]

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
    results_nb = execute_naive_bayes(x_train_tfidf, y_train, k_folds)

    results_lr = execute_logistic_regression(x_train_tfidf, y_train, k_folds)

    results_svm = execute_SVM(x_train_tfidf, y_train, k_folds)

    results_rf = execute_random_forest(x_train_tfidf, y_train, k_folds)

    # Coletando metricas [acuracia, precisao, revocacao]
    metrics_nb = [accuracy_score(y_train, results_nb),
                  precision_score(y_train, results_nb), 
                  recall_score(y_train, results_nb)]
    
    metrics_lr = [accuracy_score(y_train, results_lr),
                    precision_score(y_train, results_lr),
                    recall_score(y_train, results_lr)]
    
    metrics_svm = [accuracy_score(y_train, results_svm),
                    precision_score(y_train, results_svm),
                    recall_score(y_train, results_svm)]
    
    metrics_rf = [accuracy_score(y_train, results_rf),
                    precision_score(y_train, results_rf),
                    recall_score(y_train, results_rf)]
    
    print('\n\n** K-folds: ', k_folds)

    # Printando metricas
    print('\nNaive Bayes')
    print('Acuracia: ', metrics_nb[0])
    print('Precisao: ', metrics_nb[1])
    print('Revocacao: ', metrics_nb[2])

    print('\nLogistic Regression')
    print('Acuracia: ', metrics_lr[0])
    print('Precisao: ', metrics_lr[1])
    print('Revocacao: ', metrics_lr[2])

    print('\nSVM')
    print('Acuracia: ', metrics_svm[0])
    print('Precisao: ', metrics_svm[1])
    print('Revocacao: ', metrics_svm[2])

    print('\nRandom Forest')
    print('Acuracia: ', metrics_rf[0])
    print('Precisao: ', metrics_rf[1])
    print('Revocacao: ', metrics_rf[2])

    # Matriz de confusao
    print('\nMatrizes confusao')
    print('Naive Bayes')
    print(pd.crosstab(y_train, results_nb, rownames=['Real'], colnames=['Predito'], margins=True))
    print('\nLogistic Regression')
    print(pd.crosstab(y_train, results_lr, rownames=['Real'], colnames=['Predito'], margins=True))
    print('\nSVM')
    print(pd.crosstab(y_train, results_svm, rownames=['Real'], colnames=['Predito'], margins=True))
    print('\nRandom Forest')
    print(pd.crosstab(y_train, results_rf, rownames=['Real'], colnames=['Predito'], margins=True))