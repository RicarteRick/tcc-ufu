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
    # CountVectorizer para frequencia
    vectorizer = CountVectorizer(max_features=1473)
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
    # print('\n\n** Prediction: Hold-out: ' + str(100-test_size*100) + ' - ' + str(test_size*100))
    
    # print('\nNaive Bayes')
    # print('Acuracia: ', metrics_nb[0])
    # print('Precisao: ', metrics_nb[1])
    # print('Revocacao: ', metrics_nb[2])
    # print('Matriz de confusao: ')
    # print(metrics_nb[3])

    # print('\nLogistic Regression')
    # print('Acuracia: ', metrics_lr[0])
    # print('Precisao: ', metrics_lr[1])
    # print('Revocacao: ', metrics_lr[2])
    # print('Matriz de confusao: ')
    # print(metrics_lr[3])

    # print('\nSVM')
    # print('Acuracia: ', metrics_svm[0])
    # print('Precisao: ', metrics_svm[1])
    # print('Revocacao: ', metrics_svm[2])
    # print('Matriz de confusao: ')
    # print(metrics_svm[3])

    # print('\nRandom Forest')
    # print('Acuracia: ', metrics_rf[0])
    # print('Precisao: ', metrics_rf[1])
    # print('Revocacao: ', metrics_rf[2])
    # print('Matriz de confusao: ')
    # print(metrics_rf[3])

    return [metrics_nb, metrics_lr, metrics_svm, metrics_rf], nb_classifier, lr_classifier, svm_classifier, rf_classifier

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
    # print('\n\n** CV: K-folds: ', k_folds)

    # print('\nNaive Bayes')
    # print('Acuracia: ', metrics_nb[0])
    # print('Precisao: ', metrics_nb[1])
    # print('Revocacao: ', metrics_nb[2])
    # print('Matriz de confusao: ')
    # print(metrics_nb[3])

    # print('\nLogistic Regression')
    # print('Acuracia: ', metrics_lr[0])
    # print('Precisao: ', metrics_lr[1])
    # print('Revocacao: ', metrics_lr[2])
    # print('Matriz de confusao: ')
    # print(metrics_lr[3])

    # print('\nSVM')
    # print('Acuracia: ', metrics_svm[0])
    # print('Precisao: ', metrics_svm[1])
    # print('Revocacao: ', metrics_svm[2])
    # print('Matriz de confusao: ')
    # print(metrics_svm[3])

    # print('\nRandom Forest')
    # print('Acuracia: ', metrics_rf[0])
    # print('Precisao: ', metrics_rf[1])
    # print('Revocacao: ', metrics_rf[2])
    # print('Matriz de confusao: ')
    # print(metrics_rf[3])

    return [metrics_nb, metrics_lr, metrics_svm, metrics_rf], nb_classifier, lr_classifier, svm_classifier, rf_classifier

def execute_classifiers(filePath):
    # Carregar o arquivo JSON em um DataFrame
    data = pd.read_json(filePath)

    #region variaveis
    predict_train_test_metrics = []
    train_test_classifiers = []
    nb_classifiers = []
    lr_classifiers = []
    svm_classifiers = []
    rf_classifiers = []

    predict_cross_validation_metrics = []
    cross_validation_classifiers = []
    nb_cv_classifiers = []
    lr_cv_classifiers = []
    svm_cv_classifiers = []
    rf_cv_classifiers = []
    #endregion

    # Predicao train x test
    predict_70_30_metrics, nb_classifier_70_30, lr_classifier_70_30, svm_classifier_70_30, rf_classifier_70_30 = prediction(data, 0.3) # 70-30
    predict_80_20_metrics, nb_classifier_80_20, lr_classifier_80_20, svm_classifier_80_20, rf_classifier_80_20 = prediction(data, 0.2) # 80-20
    predict_90_10_metrics, nb_classifier_90_10, lr_classifier_90_10, svm_classifier_90_10, rf_classifier_90_10 = prediction(data, 0.1) # 90-10

    #region Salvando metricas e classificadores
    predict_train_test_metrics.append(predict_70_30_metrics)
    predict_train_test_metrics.append(predict_80_20_metrics)
    predict_train_test_metrics.append(predict_90_10_metrics)

    nb_classifiers.append(nb_classifier_70_30)
    nb_classifiers.append(nb_classifier_80_20)
    nb_classifiers.append(nb_classifier_90_10)

    lr_classifiers.append(lr_classifier_70_30)
    lr_classifiers.append(lr_classifier_80_20)
    lr_classifiers.append(lr_classifier_90_10)

    svm_classifiers.append(svm_classifier_70_30)
    svm_classifiers.append(svm_classifier_80_20)
    svm_classifiers.append(svm_classifier_90_10)

    rf_classifiers.append(rf_classifier_70_30)
    rf_classifiers.append(rf_classifier_80_20)
    rf_classifiers.append(rf_classifier_90_10)

    train_test_classifiers.append(nb_classifiers)
    train_test_classifiers.append(lr_classifiers)
    train_test_classifiers.append(svm_classifiers)
    train_test_classifiers.append(rf_classifiers)
    #endregion

    # Validacao cruzada
    cv_3_folds_metrics, nb_classifier_cv_3, lr_classifier_cv_3, svm_classifier_cv_3, rf_classifier_cv_3 = cross_validation(data, 3)
    cv_5_folds_metrics, nb_classifier_cv_5, lr_classifier_cv_5, svm_classifier_cv_5, rf_classifier_cv_5 = cross_validation(data, 5)
    cv_10_folds_metrics, nb_classifier_cv_10, lr_classifier_cv_10, svm_classifier_cv_10, rf_classifier_cv_10 = cross_validation(data, 10)

    #region Salvando metricas e classificadores CV
    predict_cross_validation_metrics.append(cv_3_folds_metrics)
    predict_cross_validation_metrics.append(cv_5_folds_metrics)
    predict_cross_validation_metrics.append(cv_10_folds_metrics)

    nb_cv_classifiers.append(nb_classifier_cv_3)
    nb_cv_classifiers.append(nb_classifier_cv_5)
    nb_cv_classifiers.append(nb_classifier_cv_10)

    lr_cv_classifiers.append(lr_classifier_cv_3)
    lr_cv_classifiers.append(lr_classifier_cv_5)
    lr_cv_classifiers.append(lr_classifier_cv_10)

    svm_cv_classifiers.append(svm_classifier_cv_3)
    svm_cv_classifiers.append(svm_classifier_cv_5)
    svm_cv_classifiers.append(svm_classifier_cv_10)
    
    rf_cv_classifiers.append(rf_classifier_cv_3)
    rf_cv_classifiers.append(rf_classifier_cv_5)
    rf_cv_classifiers.append(rf_classifier_cv_10)

    cross_validation_classifiers.append(nb_cv_classifiers)
    cross_validation_classifiers.append(lr_cv_classifiers)
    cross_validation_classifiers.append(svm_cv_classifiers)
    cross_validation_classifiers.append(rf_cv_classifiers)
    #endregion

    return predict_train_test_metrics, predict_cross_validation_metrics, train_test_classifiers, cross_validation_classifiers

def test_data(filePath, train_test_classifiers, cross_validation_classifiers):
    data = pd.read_json(filePath)

    results_nb = []
    results_lr = []
    results_svm = []
    results_rf = []

    results_cv_nb = []
    results_cv_lr = []
    results_cv_svm = []
    results_cv_rf = []

    #region Extraindo variaveis
    nb_classifiers = train_test_classifiers[0]
    lr_classifiers = train_test_classifiers[1]
    svm_classifiers = train_test_classifiers[2]
    rf_classifiers = train_test_classifiers[3]

    nb_cv_classifiers = cross_validation_classifiers[0]
    lr_cv_classifiers = cross_validation_classifiers[1]
    svm_cv_classifiers = cross_validation_classifiers[2]
    rf_cv_classifiers = cross_validation_classifiers[3]
    #endregion

    all_text_data = data['TweetContent'].apply(lambda tweet: ' '.join(tweet))
    all_data_tfidf = vetorize_data(all_text_data)

    # Aplicar predicao nos classificadores treinados
    #region Predicao train x test
    results_nb_70_30 = nb_classifiers[0].predict(all_data_tfidf)
    results_nb_80_20 = nb_classifiers[1].predict(all_data_tfidf)
    results_nb_90_10 = nb_classifiers[2].predict(all_data_tfidf)

    results_lr_70_30 = lr_classifiers[0].predict(all_data_tfidf)
    results_lr_80_20 = lr_classifiers[1].predict(all_data_tfidf)
    results_lr_90_10 = lr_classifiers[2].predict(all_data_tfidf)

    results_svm_70_30 = svm_classifiers[0].predict(all_data_tfidf)
    results_svm_80_20 = svm_classifiers[1].predict(all_data_tfidf)
    results_svm_90_10 = svm_classifiers[2].predict(all_data_tfidf)

    results_rf_70_30 = rf_classifiers[0].predict(all_data_tfidf)
    results_rf_80_20 = rf_classifiers[1].predict(all_data_tfidf)
    results_rf_90_10 = rf_classifiers[2].predict(all_data_tfidf)

    results_nb.append(results_nb_70_30)
    results_nb.append(results_nb_80_20)
    results_nb.append(results_nb_90_10)

    results_lr.append(results_lr_70_30)
    results_lr.append(results_lr_80_20)
    results_lr.append(results_lr_90_10)

    results_svm.append(results_svm_70_30)
    results_svm.append(results_svm_80_20)
    results_svm.append(results_svm_90_10)

    results_rf.append(results_rf_70_30)
    results_rf.append(results_rf_80_20)
    results_rf.append(results_rf_90_10)
    #endregion

    #region Predicao cross validation
    results_nb_cv_3 = nb_cv_classifiers[0].predict(all_data_tfidf)
    results_nb_cv_5 = nb_cv_classifiers[1].predict(all_data_tfidf)
    results_nb_cv_10 = nb_cv_classifiers[2].predict(all_data_tfidf)

    results_lr_cv_3 = lr_cv_classifiers[0].predict(all_data_tfidf)
    results_lr_cv_5 = lr_cv_classifiers[1].predict(all_data_tfidf)
    results_lr_cv_10 = lr_cv_classifiers[2].predict(all_data_tfidf)

    results_svm_cv_3 = svm_cv_classifiers[0].predict(all_data_tfidf)
    results_svm_cv_5 = svm_cv_classifiers[1].predict(all_data_tfidf)
    results_svm_cv_10 = svm_cv_classifiers[2].predict(all_data_tfidf)

    results_rf_cv_3 = rf_cv_classifiers[0].predict(all_data_tfidf)
    results_rf_cv_5 = rf_cv_classifiers[1].predict(all_data_tfidf)
    results_rf_cv_10 = rf_cv_classifiers[2].predict(all_data_tfidf)

    results_cv_nb.append(results_nb_cv_3)
    results_cv_nb.append(results_nb_cv_5)
    results_cv_nb.append(results_nb_cv_10)

    results_cv_lr.append(results_lr_cv_3)
    results_cv_lr.append(results_lr_cv_5)
    results_cv_lr.append(results_lr_cv_10)

    results_cv_svm.append(results_svm_cv_3)
    results_cv_svm.append(results_svm_cv_5)
    results_cv_svm.append(results_svm_cv_10)

    results_cv_rf.append(results_rf_cv_3)
    results_cv_rf.append(results_rf_cv_5)
    results_cv_rf.append(results_rf_cv_10)
    #endregion

    return results_nb, results_lr, results_svm, results_rf, results_cv_nb, results_cv_lr, results_cv_svm, results_cv_rf, all_text_data
