from pre_processamento import preprocess
from classificadores import execute_classifiers, test_data
from salvar_resultados import save_results_train_test, save_results_cross_validation, save_predicts_train_test, save_predicts_cross_validation

# Paths
docs_path = 'docs/'
results_path = 'results/'
results_train_test_path = results_path+'train_test/'
results_cross_validation_path = results_path+'cross_validation/'
predicts_train_test_path = results_train_test_path + 'predicts/'
predicts_cross_validation_path = results_cross_validation_path + 'predicts/'

raw_file_name = 'base_de_treino.json'
preprocessed_file_name = 'base_de_treino_pre_processada.json'
tokenized_raw_file_name = 'base_de_treino_raw_tokenizada.json'

raw_test_file_name = 'geral_sem_duplicatas.json'
preprocessed_test_file_name = 'base_de_teste_pre_processada.json'
tokenized_raw_test_file_name = 'base_de_teste_raw_tokenizada.json'

# Pre-processamento
# preprocess(docs_path, raw_file_name, preprocessed_file_name, tokenized_raw_file_name)
# preprocess(docs_path, raw_test_file_name, preprocessed_test_file_name, tokenized_raw_test_file_name)

# Classificadores
# ** predict_train_test_metrics -> [70-30, 80-20, 90-10]
# ** predict_cross_validation_metrics -> [3, 5, 10]
# ** train_test_classifiers -> [nb[70-30, 80-20, 90-10], lr[70-30, 80-20, 90-10], svm[70-30, 80-20, 90-10], rf[70-30, 80-20, 90-10]]
# ** cross_validation_classifiers -> [nb[3, 5, 10], lr[3, 5, 10], svm[3, 5, 10], rf[3, 5, 10]]
predict_train_test_metrics, predict_cross_validation_metrics, train_test_classifiers, cross_validation_classifiers = execute_classifiers(docs_path+preprocessed_file_name)

# Salvando resultados (metricas, matriz de confusao)
save_results_train_test(results_train_test_path, predict_train_test_metrics)
save_results_cross_validation(results_cross_validation_path, predict_cross_validation_metrics)

# Testando classificadores com base de teste
results_nb, results_lr, results_svm, results_rf, results_cv_nb, results_cv_lr, results_cv_svm, results_cv_rf, all_text_test_data = test_data(docs_path+preprocessed_test_file_name, train_test_classifiers, cross_validation_classifiers)

# Salvando predicoes da base de teste
save_predicts_train_test(results_nb, results_lr, results_svm, results_rf, all_text_test_data, predicts_train_test_path)
save_predicts_cross_validation(results_cv_nb, results_cv_lr, results_cv_svm, results_cv_rf, all_text_test_data, predicts_cross_validation_path)