from pre_processamento import preprocess
from classificadores import execute_classifiers
from salvar_resultados import save_results_train_test, save_results_cross_validation

# Paths
docs_path = 'docs/'
results_path = 'results/'
raw_file_name = 'base_de_treino.json'
preprocessed_file_name = 'base_de_treino_pre_processada.json'
tokenized_raw_file_name = 'base_de_treino_raw_tokenizada.json'
results_train_test_path = results_path+'train_test/'
results_cross_validation_path = results_path+'cross_validation/'

# Pre-processamento
preprocess(docs_path, raw_file_name, preprocessed_file_name, tokenized_raw_file_name)

# Classificadores
predict_70_30_metrics, predict_80_20_metrics, predict_90_10_metrics, cv_3_folds_metrics, cv_5_folds_metrics, cv_10_folds_metrics = execute_classifiers(docs_path+preprocessed_file_name)

# Salvando resultados
save_results_train_test(results_train_test_path, predict_70_30_metrics, predict_80_20_metrics, predict_90_10_metrics)
save_results_cross_validation(results_cross_validation_path, cv_3_folds_metrics, cv_5_folds_metrics, cv_10_folds_metrics)