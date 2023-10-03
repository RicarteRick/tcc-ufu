import pandas as pd

header_line = f"Classificador;Acuracia;Precisao;Revocacao\n"
matrix_line = f"Matriz de confusao\n"

def save_results_train_test(results_train_test_path, predict_train_test_metrics):
    with open(results_train_test_path+'70_30.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_train_test_metrics[0][0][0]};{predict_train_test_metrics[0][0][1]};{predict_train_test_metrics[0][0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[0][0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_train_test_metrics[0][1][0]};{predict_train_test_metrics[0][1][1]};{predict_train_test_metrics[0][1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[0][1][3]))

        output_file.write(f"\n\nSVM;{predict_train_test_metrics[0][2][0]};{predict_train_test_metrics[0][2][1]};{predict_train_test_metrics[0][2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[0][2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_train_test_metrics[0][3][0]};{predict_train_test_metrics[0][3][1]};{predict_train_test_metrics[0][3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[0][3][3]))

    with open(results_train_test_path+'80_20.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_train_test_metrics[1][0][0]};{predict_train_test_metrics[1][0][1]};{predict_train_test_metrics[1][0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[1][0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_train_test_metrics[1][1][0]};{predict_train_test_metrics[1][1][1]};{predict_train_test_metrics[1][1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[1][1][3]))

        output_file.write(f"\n\nSVM;{predict_train_test_metrics[1][2][0]};{predict_train_test_metrics[1][2][1]};{predict_train_test_metrics[1][2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[1][2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_train_test_metrics[1][3][0]};{predict_train_test_metrics[1][3][1]};{predict_train_test_metrics[1][3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[1][3][3]))

    with open(results_train_test_path+'90_10.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_train_test_metrics[2][0][0]};{predict_train_test_metrics[2][0][1]};{predict_train_test_metrics[2][0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[2][0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_train_test_metrics[2][1][0]};{predict_train_test_metrics[2][1][1]};{predict_train_test_metrics[2][1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[2][1][3]))

        output_file.write(f"\n\nSVM;{predict_train_test_metrics[2][2][0]};{predict_train_test_metrics[2][2][1]};{predict_train_test_metrics[2][2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[2][2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_train_test_metrics[2][3][0]};{predict_train_test_metrics[2][3][1]};{predict_train_test_metrics[2][3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_train_test_metrics[2][3][3]))

def save_results_cross_validation(results_cross_validation_path, predict_cross_validation_metrics):
    with open(results_cross_validation_path+'3_folds.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_cross_validation_metrics[0][0][0]};{predict_cross_validation_metrics[0][0][1]};{predict_cross_validation_metrics[0][0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[0][0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_cross_validation_metrics[0][1][0]};{predict_cross_validation_metrics[0][1][1]};{predict_cross_validation_metrics[0][1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[0][1][3]))

        output_file.write(f"\n\nSVM;{predict_cross_validation_metrics[0][2][0]};{predict_cross_validation_metrics[0][2][1]};{predict_cross_validation_metrics[0][2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[0][2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_cross_validation_metrics[0][3][0]};{predict_cross_validation_metrics[0][3][1]};{predict_cross_validation_metrics[0][3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[0][3][3]))

    with open(results_cross_validation_path+'5_folds.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_cross_validation_metrics[1][0][0]};{predict_cross_validation_metrics[1][0][1]};{predict_cross_validation_metrics[1][0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[1][0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_cross_validation_metrics[1][1][0]};{predict_cross_validation_metrics[1][1][1]};{predict_cross_validation_metrics[1][1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[1][1][3]))

        output_file.write(f"\n\nSVM;{predict_cross_validation_metrics[1][2][0]};{predict_cross_validation_metrics[1][2][1]};{predict_cross_validation_metrics[1][2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[1][2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_cross_validation_metrics[1][3][0]};{predict_cross_validation_metrics[1][3][1]};{predict_cross_validation_metrics[1][3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[1][3][3]))

    with open(results_cross_validation_path+'10_folds.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_cross_validation_metrics[2][0][0]};{predict_cross_validation_metrics[2][0][1]};{predict_cross_validation_metrics[2][0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[2][0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_cross_validation_metrics[2][1][0]};{predict_cross_validation_metrics[2][1][1]};{predict_cross_validation_metrics[2][1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[2][1][3]))

        output_file.write(f"\n\nSVM;{predict_cross_validation_metrics[2][2][0]};{predict_cross_validation_metrics[2][2][1]};{predict_cross_validation_metrics[2][2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[2][2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_cross_validation_metrics[2][3][0]};{predict_cross_validation_metrics[2][3][1]};{predict_cross_validation_metrics[2][3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_cross_validation_metrics[2][3][3]))

def save_predicts_train_test(results_nb, results_lr, results_svm, results_rf, all_text_data, predicts_train_test_path):
    predicts_nb_70_30_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_nb[0]})
    predicts_nb_80_20_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_nb[1]})
    predicts_nb_90_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_nb[2]})

    predicts_lr_70_30_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_lr[0]})
    predicts_lr_80_20_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_lr[1]})
    predicts_lr_90_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_lr[2]})

    predicts_svm_70_30_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_svm[0]})
    predicts_svm_80_20_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_svm[1]})
    predicts_svm_90_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_svm[2]})

    predicts_rf_70_30_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_rf[0]})
    predicts_rf_80_20_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_rf[1]})
    predicts_rf_90_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_rf[2]})

    predicts_nb_70_30_df.to_csv(predicts_train_test_path+'nb_70_30.csv', index=False)
    predicts_nb_80_20_df.to_csv(predicts_train_test_path+'nb_80_20.csv', index=False)
    predicts_nb_90_10_df.to_csv(predicts_train_test_path+'nb_90_10.csv', index=False)

    predicts_lr_70_30_df.to_csv(predicts_train_test_path+'lr_70_30.csv', index=False)
    predicts_lr_80_20_df.to_csv(predicts_train_test_path+'lr_80_20.csv', index=False)
    predicts_lr_90_10_df.to_csv(predicts_train_test_path+'lr_90_10.csv', index=False)

    predicts_svm_70_30_df.to_csv(predicts_train_test_path+'svm_70_30.csv', index=False)
    predicts_svm_80_20_df.to_csv(predicts_train_test_path+'svm_80_20.csv', index=False)
    predicts_svm_90_10_df.to_csv(predicts_train_test_path+'svm_90_10.csv', index=False)

    predicts_rf_70_30_df.to_csv(predicts_train_test_path+'rf_70_30.csv', index=False)
    predicts_rf_80_20_df.to_csv(predicts_train_test_path+'rf_80_20.csv', index=False)
    predicts_rf_90_10_df.to_csv(predicts_train_test_path+'rf_90_10.csv', index=False)

def save_predicts_cross_validation(results_cv_nb, results_cv_lr, results_cv_svm, results_cv_rf, all_text_data, predicts_cross_validation_path):
    predicts_cv_nb_3_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_nb[0]})
    predicts_cv_nb_5_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_nb[1]})
    predicts_cv_nb_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_nb[2]})

    predicts_cv_lr_3_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_lr[0]})
    predicts_cv_lr_5_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_lr[1]})
    predicts_cv_lr_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_lr[2]})

    predicts_cv_svm_3_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_svm[0]})
    predicts_cv_svm_5_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_svm[1]})
    predicts_cv_svm_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_svm[2]})

    predicts_cv_rf_3_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_rf[0]})
    predicts_cv_rf_5_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_rf[1]})
    predicts_cv_rf_10_df = pd.DataFrame({'Content': all_text_data, 'IsRelated': results_cv_rf[2]})

    predicts_cv_nb_3_df.to_csv(predicts_cross_validation_path+'nb_3.csv', index=False)
    predicts_cv_nb_5_df.to_csv(predicts_cross_validation_path+'nb_5.csv', index=False)
    predicts_cv_nb_10_df.to_csv(predicts_cross_validation_path+'nb_10.csv', index=False)

    predicts_cv_lr_3_df.to_csv(predicts_cross_validation_path+'lr_3.csv', index=False)
    predicts_cv_lr_5_df.to_csv(predicts_cross_validation_path+'lr_5.csv', index=False)
    predicts_cv_lr_10_df.to_csv(predicts_cross_validation_path+'lr_10.csv', index=False)

    predicts_cv_svm_3_df.to_csv(predicts_cross_validation_path+'svm_3.csv', index=False)
    predicts_cv_svm_5_df.to_csv(predicts_cross_validation_path+'svm_5.csv', index=False)
    predicts_cv_svm_10_df.to_csv(predicts_cross_validation_path+'svm_10.csv', index=False)

    predicts_cv_rf_3_df.to_csv(predicts_cross_validation_path+'rf_3.csv', index=False)
    predicts_cv_rf_5_df.to_csv(predicts_cross_validation_path+'rf_5.csv', index=False)
    predicts_cv_rf_10_df.to_csv(predicts_cross_validation_path+'rf_10.csv', index=False)