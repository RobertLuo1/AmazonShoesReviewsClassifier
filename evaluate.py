#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,classification_report,confusion_matrix,roc_curve,auc,precision_score,recall_score
import config
import torch
import seaborn as sns
from textdataset import getdataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def evaluate(device):
    label_all = []
    pred_all = []
    pred_prob = []
    test_dataloader = getdataLoader("test")
    model = torch.load(config.model_file)
    model.eval()
    for text,label,input_length in test_dataloader:
        with torch.no_grad():
            text,label = text.to(device),label.to(device)
            out = model(text,input_length)
            pred_pro = F.softmax(out,dim=1)
            pred_pro = pred_pro[:,-1].cpu().numpy()
            pred = out.max(dim=-1)[-1]
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
            pred_all.extend(pred)
            label_all.extend(label)
            pred_prob.extend(pred_pro)

    return (pred_all,label_all,pred_prob)

def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob,pos_label=1)
    roc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig('./picture/roc_curve_neural_network')

def confusionMatrix(confusion):
    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, ax=ax)
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()

if __name__ == '__main__':
    pred_all,label_all,pred_prob = evaluate(config.device)
    f1Score = f1_score(label_all, pred_all)
    precision = precision_score(label_all,pred_all)
    recall = recall_score(label_all,pred_all)
    # auc = roc_auc_score(label_all, pred_all)
    acc = accuracy_score(label_all, pred_all)
    confusion = confusion_matrix(label_all, pred_all)
    # plot_roc(label_all,pred_prob)
    # report = classification_report(label_all, pred_all, target_names=['negative', 'positive'], digits=2)
    # confusionMatrix(confusion)
    print(f'f1 score is {f1Score:.2f}',f'precision is {precision:.2f}',f'accuracy is {acc:.2f}',f'recall is {recall}')
    # print(report)

