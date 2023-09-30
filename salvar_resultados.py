header_line = f"Classificador;Acuracia;Precisao;Revocacao\n"
matrix_line = f"Matriz de confusao\n"

def save_results_train_test(results_train_test_path, predict_70_30_metrics, predict_80_20_metrics, predict_90_10_metrics):
    with open(results_train_test_path+'70_30.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_70_30_metrics[0][0]};{predict_70_30_metrics[0][1]};{predict_70_30_metrics[0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_70_30_metrics[0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_70_30_metrics[1][0]};{predict_70_30_metrics[1][1]};{predict_70_30_metrics[1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_70_30_metrics[1][3]))

        output_file.write(f"\n\nSVM;{predict_70_30_metrics[2][0]};{predict_70_30_metrics[2][1]};{predict_70_30_metrics[2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_70_30_metrics[2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_70_30_metrics[3][0]};{predict_70_30_metrics[3][1]};{predict_70_30_metrics[3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_70_30_metrics[3][3]))

    with open(results_train_test_path+'80_20.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_80_20_metrics[0][0]};{predict_80_20_metrics[0][1]};{predict_80_20_metrics[0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_80_20_metrics[0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_80_20_metrics[1][0]};{predict_80_20_metrics[1][1]};{predict_80_20_metrics[1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_80_20_metrics[1][3]))

        output_file.write(f"\n\nSVM;{predict_80_20_metrics[2][0]};{predict_80_20_metrics[2][1]};{predict_80_20_metrics[2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_80_20_metrics[2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_80_20_metrics[3][0]};{predict_80_20_metrics[3][1]};{predict_80_20_metrics[3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_80_20_metrics[3][3]))

    with open(results_train_test_path+'90_10.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{predict_90_10_metrics[0][0]};{predict_90_10_metrics[0][1]};{predict_90_10_metrics[0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_90_10_metrics[0][3]))

        output_file.write(f"\n\nLogistic Regression;{predict_90_10_metrics[1][0]};{predict_90_10_metrics[1][1]};{predict_90_10_metrics[1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_90_10_metrics[1][3]))

        output_file.write(f"\n\nSVM;{predict_90_10_metrics[2][0]};{predict_90_10_metrics[2][1]};{predict_90_10_metrics[2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_90_10_metrics[2][3]))

        output_file.write(f"\n\nRandom Forest;{predict_90_10_metrics[3][0]};{predict_90_10_metrics[3][1]};{predict_90_10_metrics[3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(predict_90_10_metrics[3][3]))

def save_results_cross_validation(results_cross_validation_path, cv_3_folds_metrics, cv_5_folds_metrics, cv_10_folds_metrics):
    with open(results_cross_validation_path+'3_folds.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{cv_3_folds_metrics[0][0]};{cv_3_folds_metrics[0][1]};{cv_3_folds_metrics[0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_3_folds_metrics[0][3]))

        output_file.write(f"\n\nLogistic Regression;{cv_3_folds_metrics[1][0]};{cv_3_folds_metrics[1][1]};{cv_3_folds_metrics[1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_3_folds_metrics[1][3]))

        output_file.write(f"\n\nSVM;{cv_3_folds_metrics[2][0]};{cv_3_folds_metrics[2][1]};{cv_3_folds_metrics[2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_3_folds_metrics[2][3]))

        output_file.write(f"\n\nRandom Forest;{cv_3_folds_metrics[3][0]};{cv_3_folds_metrics[3][1]};{cv_3_folds_metrics[3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_3_folds_metrics[3][3]))

    with open(results_cross_validation_path+'5_folds.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{cv_5_folds_metrics[0][0]};{cv_5_folds_metrics[0][1]};{cv_5_folds_metrics[0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_5_folds_metrics[0][3]))

        output_file.write(f"\n\nLogistic Regression;{cv_5_folds_metrics[1][0]};{cv_5_folds_metrics[1][1]};{cv_5_folds_metrics[1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_5_folds_metrics[1][3]))

        output_file.write(f"\n\nSVM;{cv_5_folds_metrics[2][0]};{cv_5_folds_metrics[2][1]};{cv_5_folds_metrics[2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_5_folds_metrics[2][3]))

        output_file.write(f"\n\nRandom Forest;{cv_5_folds_metrics[3][0]};{cv_5_folds_metrics[3][1]};{cv_5_folds_metrics[3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_5_folds_metrics[3][3]))

    with open(results_cross_validation_path+'10_folds.out', 'w') as output_file:
        output_file.write(header_line)
        output_file.write(f"Naive Bayes;{cv_10_folds_metrics[0][0]};{cv_10_folds_metrics[0][1]};{cv_10_folds_metrics[0][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_10_folds_metrics[0][3]))

        output_file.write(f"\n\nLogistic Regression;{cv_10_folds_metrics[1][0]};{cv_10_folds_metrics[1][1]};{cv_10_folds_metrics[1][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_10_folds_metrics[1][3]))

        output_file.write(f"\n\nSVM;{cv_10_folds_metrics[2][0]};{cv_10_folds_metrics[2][1]};{cv_10_folds_metrics[2][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_10_folds_metrics[2][3]))

        output_file.write(f"\n\nRandom Forest;{cv_10_folds_metrics[3][0]};{cv_10_folds_metrics[3][1]};{cv_10_folds_metrics[3][2]}\n")
        output_file.write(matrix_line)
        output_file.write(str(cv_10_folds_metrics[3][3]))