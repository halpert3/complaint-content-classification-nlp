import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, accuracy_score, plot_confusion_matrix


def scoring(y_pred_from_X_data, y_true_data, model, X_data):
    # Calculates and prints scores for the model
    accuracy = accuracy_score(y_true_data, y_pred_from_X_data)
    precision = precision_score(y_true_data, y_pred_from_X_data, zero_division=0)
    recall = recall_score(y_true_data, y_pred_from_X_data, zero_division=0)
    f1 = f1_score(y_true_data, y_pred_from_X_data)
    conf = confusion_matrix(y_true_data, y_pred_from_X_data)

    print("Accuracy: {:.1%}".format(accuracy))
    print("Precision: {:.1%}".format(precision))
    print("Recall: {:.1%}".format(recall))
    print("F1: {:.1%}".format(f1))
    print("Conufusion Matrix: ")
    print(conf)
    print('\n')
    
    # Plots a confusion matrix graphic (defined below)
    plot = plot_c_matrix(model, X_data, y_true_data)

    
def plot_c_matrix(model, X_data, y_data):
    # Generates a confusion matrix graphic
    plot_confusion_matrix(model, X_data, y_data,
                      display_labels=["not serious", "serious"],
                      values_format=".5g")
    plt.grid(False)
    plt.show()

def add_scores_to_scoring_df(model_name, y_test, predictions_from_X_test, y_train_balanced, predictions_from_X_train):
    
    # Adds scores to a dataframe for easy comparison
    model_scores = pd.read_csv('../data/model_scores.csv', index_col=0)
    test_accuracy = round(accuracy_score(y_test, predictions_from_X_test) * 100, 1)
    test_precision = round(precision_score(y_test, predictions_from_X_test, zero_division=0) * 100, 1)
    test_recall = round(recall_score(y_test, predictions_from_X_test, zero_division=0) * 100, 1)
    test_f1 = round(f1_score(y_test, predictions_from_X_test) * 100, 1)

    train_accuracy = round(accuracy_score(y_train_balanced, predictions_from_X_train) * 100, 1)
    train_precision = round(precision_score(y_train_balanced, predictions_from_X_train, zero_division=0) * 100, 1)
    train_recall = round(recall_score(y_train_balanced, predictions_from_X_train, zero_division=0) * 100, 1)
    train_f1 = round(f1_score(y_train_balanced, predictions_from_X_train) * 100, 1)
    
    scores = {'model': model_name,
          'test_accuracy': test_accuracy,
          'test_precision': test_precision,
          'test_recall': test_recall,
          'test_f1': test_f1, 'train_accuracy': train_accuracy, 
          'train_precision': train_precision, 
          'train_recall': train_recall,
          'train_f1': train_f1}
    
    model_scores = model_scores.append(scores, ignore_index=True)
    model_scores.drop_duplicates(keep="last", inplace=True)
    model_scores.to_csv('../data/model_scores.csv')
    pass
 
def add_scores_to_df(model_name, df, y_test, predictions_from_X_test, y_train_balanced, predictions_from_X_train):
    # Adds scores to a dataframe for easy comparison

    model_scores = df
 
    test_accuracy = round(accuracy_score(y_test, predictions_from_X_test) * 100, 3)
    test_precision = round(precision_score(y_test, predictions_from_X_test, zero_division=0) * 100, 3)
    test_recall = round(recall_score(y_test, predictions_from_X_test, zero_division=0) * 100, 3)
    test_f1 = round(f1_score(y_test, predictions_from_X_test) * 100, 3)

    train_accuracy = round(accuracy_score(y_train_balanced, predictions_from_X_train) * 100, 3)
    train_precision = round(precision_score(y_train_balanced, predictions_from_X_train, zero_division=0) * 100, 3)
    train_recall = round(recall_score(y_train_balanced, predictions_from_X_train, zero_division=0) * 100, 3)
    train_f1 = round(f1_score(y_train_balanced, predictions_from_X_train) * 100, 3)
    
    scores = {'model': model_name,
          'test_accuracy': test_accuracy,
          'test_precision': test_precision,
          'test_recall': test_recall,
          'test_f1': test_f1, 'train_accuracy': train_accuracy, 
          'train_precision': train_precision, 
          'train_recall': train_recall,
          'train_f1': train_f1}
    df_temp = pd.DataFrame.from_records(scores, index=[0])
#     columns = ['model', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'train_accuracy', 'train_precision', 'train_recall']

    new_df = pd.concat([model_scores, df_temp], sort=False)
#     model_scores = model_scores.append(scores, ignore_index=True)
#     model_scores.drop_duplicates(keep="last", inplace=True)
    new_df = new_df[['model', 'train_accuracy', 'test_accuracy', 'train_precision', 'test_precision', 'train_recall', 'test_recall', 'train_f1', 'test_f1']]
    new_df.set_index('model')
    return new_df