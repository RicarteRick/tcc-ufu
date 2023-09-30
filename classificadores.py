import pandas as pd
from sklearn.calibration import cross_val_predict, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#region Validacao cruzada
def train_naive_bayes(x_train, y_train):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(x_train, y_train)

    return nb_classifier

def train_logistic_regression(x_train, y_train):
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42, multi_class='multinomial')
    lr_classifier.fit(x_train, y_train)

    return lr_classifier

def train_SVM(x_train, y_train):
    svm_classifier = LinearSVC()
    svm_classifier.fit(x_train, y_train)

    return svm_classifier

def train_random_forest(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)

    return rf_classifier
#endregion

def vetorize_data(data):
    #CountVectorizer para frequencia
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(data)

    count_list = word_count_matrix.toarray().sum(axis=0)
    word_list = vectorizer.get_feature_names_out()

    word_freq = pd.DataFrame(count_list, index=word_list, columns=['Freq'])
    word_freq.sort_values(by='Freq', ascending=False).head(30)

    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    data_tfidf = tfidf_transformer.fit_transform(word_count_matrix)

    return data_tfidf

def get_metrics(y_test, results_nb, results_lr, results_svm, results_rf):
    metrics_nb = [accuracy_score(y_test, results_nb),
                  precision_score(y_test, results_nb), 
                  recall_score(y_test, results_nb)]
    
    metrics_lr = [accuracy_score(y_test, results_lr),
                    precision_score(y_test, results_lr),
                    recall_score(y_test, results_lr)]
    
    metrics_svm = [accuracy_score(y_test, results_svm),
                    precision_score(y_test, results_svm),
                    recall_score(y_test, results_svm)]
    
    metrics_rf = [accuracy_score(y_test, results_rf),
                    precision_score(y_test, results_rf),
                    recall_score(y_test, results_rf)]
    
    return [metrics_nb, metrics_lr, metrics_svm, metrics_rf]

def prediction(data, test_size):
    all_text_data = data['TweetContent'].apply(lambda tweet: ' '.join(tweet))

    all_data_tfidf = vetorize_data(all_text_data)

    x_train_tfidf, x_test_tfidf, y_train, y_test = train_test_split(all_data_tfidf, data['IsRelated'], test_size=test_size, random_state=42)

    # Executando predicao
    nb_classifier = train_naive_bayes(x_train_tfidf, y_train)
    results_nb = nb_classifier.predict(x_test_tfidf)

    lr_classifier = train_logistic_regression(x_train_tfidf, y_train)
    results_lr = lr_classifier.predict(x_test_tfidf)

    svm_classifier = train_SVM(x_train_tfidf, y_train)
    results_svm = svm_classifier.predict(x_test_tfidf)

    rf_classifier = train_random_forest(x_train_tfidf, y_train)
    results_rf = rf_classifier.predict(x_test_tfidf)

    # Calculando metricas
    metrics_nb, metrics_lr, metrics_svm, metrics_rf = get_metrics(y_test, results_nb, results_lr, results_svm, results_rf)

    confusion_matrix_nb = confusion_matrix(y_test, results_nb)
    confusion_matrix_lr = confusion_matrix(y_test, results_lr)
    confusion_matrix_svm = confusion_matrix(y_test, results_svm)
    confusion_matrix_rf = confusion_matrix(y_test, results_rf)

    metrics_nb.append(confusion_matrix_nb)
    metrics_lr.append(confusion_matrix_lr)
    metrics_svm.append(confusion_matrix_svm)
    metrics_rf.append(confusion_matrix_rf)

    # Printando metricas
    print('\n\n** Prediction: Hold-out: ' + str(100-test_size*100) + ' - ' + str(test_size*100))
    
    print('\nNaive Bayes')
    print('Acuracia: ', metrics_nb[0])
    print('Precisao: ', metrics_nb[1])
    print('Revocacao: ', metrics_nb[2])
    print('Matriz de confusao: ')
    print(metrics_nb[3])

    print('\nLogistic Regression')
    print('Acuracia: ', metrics_lr[0])
    print('Precisao: ', metrics_lr[1])
    print('Revocacao: ', metrics_lr[2])
    print('Matriz de confusao: ')
    print(metrics_lr[3])

    print('\nSVM')
    print('Acuracia: ', metrics_svm[0])
    print('Precisao: ', metrics_svm[1])
    print('Revocacao: ', metrics_svm[2])
    print('Matriz de confusao: ')
    print(metrics_svm[3])

    print('\nRandom Forest')
    print('Acuracia: ', metrics_rf[0])
    print('Precisao: ', metrics_rf[1])
    print('Revocacao: ', metrics_rf[2])
    print('Matriz de confusao: ')
    print(metrics_rf[3])

    return [metrics_nb, metrics_lr, metrics_svm, metrics_rf]

def cross_validation(data, k_folds):
    # Setando os conjuntos de dados e classes
    x_train_cv = data['TweetContent']
    y_train_cv = data['IsRelated']

    x_train_cv = [' '.join(tweet) for tweet in x_train_cv]

    # Matriz TF-IDF
    x_train_cv_tfidf = vetorize_data(x_train_cv)

    # Executando validação cruzada
    nb_classifier = train_naive_bayes(x_train_cv_tfidf, y_train_cv)
    results_nb = cross_val_predict(nb_classifier, x_train_cv_tfidf, y_train_cv, cv = k_folds)

    lr_classifier = train_logistic_regression(x_train_cv_tfidf, y_train_cv)
    results_lr = cross_val_predict(lr_classifier, x_train_cv_tfidf, y_train_cv, cv = k_folds)

    svm_classifier = train_SVM(x_train_cv_tfidf, y_train_cv)
    results_svm = cross_val_predict(svm_classifier, x_train_cv_tfidf, y_train_cv, cv = k_folds)

    rf_classifier = train_random_forest(x_train_cv_tfidf, y_train_cv)
    results_rf = cross_val_predict(rf_classifier, x_train_cv_tfidf, y_train_cv, cv = k_folds)

    # Calculando metricas
    metrics_nb, metrics_lr, metrics_svm, metrics_rf = get_metrics(y_train_cv, results_nb, results_lr, results_svm, results_rf)
    
    confusion_matrix_nb = pd.crosstab(y_train_cv, results_nb, rownames=['Real'], colnames=['Predito'], margins=True)
    confusion_matrix_lr = pd.crosstab(y_train_cv, results_lr, rownames=['Real'], colnames=['Predito'], margins=True)
    confusion_matrix_svm = pd.crosstab(y_train_cv, results_svm, rownames=['Real'], colnames=['Predito'], margins=True)
    confusion_matrix_rf = pd.crosstab(y_train_cv, results_rf, rownames=['Real'], colnames=['Predito'], margins=True)

    metrics_nb.append(confusion_matrix_nb)
    metrics_lr.append(confusion_matrix_lr)
    metrics_svm.append(confusion_matrix_svm)
    metrics_rf.append(confusion_matrix_rf)

    # Printando metricas
    print('\n\n** CV: K-folds: ', k_folds)

    print('\nNaive Bayes')
    print('Acuracia: ', metrics_nb[0])
    print('Precisao: ', metrics_nb[1])
    print('Revocacao: ', metrics_nb[2])
    print('Matriz de confusao: ')
    print(metrics_nb[3])

    print('\nLogistic Regression')
    print('Acuracia: ', metrics_lr[0])
    print('Precisao: ', metrics_lr[1])
    print('Revocacao: ', metrics_lr[2])
    print('Matriz de confusao: ')
    print(metrics_lr[3])

    print('\nSVM')
    print('Acuracia: ', metrics_svm[0])
    print('Precisao: ', metrics_svm[1])
    print('Revocacao: ', metrics_svm[2])
    print('Matriz de confusao: ')
    print(metrics_svm[3])

    print('\nRandom Forest')
    print('Acuracia: ', metrics_rf[0])
    print('Precisao: ', metrics_rf[1])
    print('Revocacao: ', metrics_rf[2])
    print('Matriz de confusao: ')
    print(metrics_rf[3])

    return metrics_nb, metrics_lr, metrics_svm, metrics_rf

def execute_classifiers(filePath):
    # Carregar o arquivo JSON em um DataFrame
    data = pd.read_json(filePath)

    # Predicao train x test
    predict_70_30_metrics = prediction(data, 0.3) # 70-30
    predict_80_20_metrics = prediction(data, 0.2) # 80-20
    predict_90_10_metrics = prediction(data, 0.1) # 90-10

    # Validacao cruzada
    cv_3_folds_metrics = cross_validation(data, 3)
    cv_5_folds_metrics = cross_validation(data, 5)
    cv_10_folds_metrics = cross_validation(data, 10)

    return predict_70_30_metrics, predict_80_20_metrics, predict_90_10_metrics, cv_3_folds_metrics, cv_5_folds_metrics, cv_10_folds_metrics