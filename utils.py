import torch
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

def categorical_accuracy(preds, y, device):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum().to(device)/torch.FloatTensor([y.shape[0]]).to(device)

def print_scores(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)

    index = labels
    columns = labels
    cm_df = pd.DataFrame(cm,columns, index)
    cm_df_p = cm_df.copy()
    cm_df_p = cm_df_p.astype(float)
    for row in index:
        for col in columns:
            cm_df_p.loc[row][col] = cm_df.loc[row][col]/(cm_df.loc[row].sum())
    plt.figure(figsize=(10,6))  
    # print(cm_df_p.head())
    sns.heatmap(cm_df_p, fmt=".1%", annot=True)
    # print(classification_report(y_test, y_pred, target_names=labels))
    
