'''
Created on Feb 1, 2017

@author: root
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import activationFunction as af
from sklearn.metrics import average_precision_score

figure = ['o', 'v', 's', 'p', '*','^', '<', '>', '8',  'h', 'H', 'D', 'd', 'P', 'X']
color = ['#DC143C','#D800FF' , '#00FF00', '#00008B' , '#FF6800' , '#800080' ,'#FF0000' , '#3CE7DA' , '#AAAAAA' ]
color2 = ['#EC7063' , '#5DADE2','#FFA07A' , '#48C9B0' , '#EB984E']

def drawROCSingle(score , class_tag , p_label=1):
    fpr, tpr , thresholds = metrics.roc_curve(class_tag , score , pos_label=p_label) 
    plt.figure()
    lw = 2
    acu = 0
    try:
        acu = metrics.auc(fpr , tpr)
    except Exception as ex:
        print(ex)
        pass
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %s)' % acu)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-',)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic- WordNet')
    plt.legend(loc="lower right")
    plt.show()

def drawCurve(x , y , x_label , y_label , curve_label, title):
    plt.figure()
    plt.plot(x, y,'bo' , color='#B01F00', label=curve_label,linewidth=1.0, linestyle="-" , dash_joinstyle = 'bevel' , markeredgecolor='#B01F00')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def draw_multiple_line_curve(data,x_label , y_label, title):
    plt.figure()
    for index, line in enumerate(data):
        x = line[0]
        y = line[1]
        label = line[2]
        plt.plot(x, y, color=color[index], marker=figure[index], linestyle="-", markersize=7, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.show()


def draw_multiple_roc(data, rel):
    plt.figure()
    for index, relation in enumerate(rel):
        pList = data[relation.label]
        ppList = data[relation.label+'_score']
        fpr, tpr, thresholds = metrics.roc_curve(pList, ppList, pos_label=1)
        auc = round(metrics.auc(fpr, tpr) , 4)
        print('relation : %s , Acc : %s' %(relation.label , auc))
        plt.subplot(3,4,index+1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='area = %s' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-',)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(relation.label)
        plt.legend(loc="lower right",prop={'size': 8})
    plt.subplots_adjust( hspace=0.5, wspace=0.4)
    plt.show()

def draw_relation_acc_bar_chart(data, rel):
    x = []
    y = []
    bar_width = 1
    y_pos = np.arange(len(rel))
    for index, relation in enumerate(rel):
        x.append(relation.label)
        y.append(data[relation.label+'_acc'])
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, np.asarray(y),align='center', color=color[0],alpha=0.4, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy')
    ax.set_title('WN11')
    plt.show()   

def draw_relation_acc_bar_chart_multiple_acc(data):

    n_groups = len(data)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2

    def_acc = [item[1] for item in data]
    acc_gns = [item[2] for item in data]
    acc_fcsa = [item[3] for item in data]
    acc = [item[4] for item in data]
    rel = [item[0] for item in data]


    plt.bar(index, def_acc, bar_width, color=color2[0], label='ENN',edgecolor = "none")
    plt.bar(index + bar_width, acc_gns, bar_width,color=color2[1], label='ENN-GNSs',edgecolor = "none")
    plt.bar(index + 2*bar_width, acc_fcsa, bar_width,color=color2[2], label='ENN-FCSA',edgecolor = "none")
    plt.bar(index + 3*bar_width, acc, bar_width,color=color2[3], label='ENN-GNSs-FCSA',edgecolor = "none")

    plt.xlabel('Relations',{'fontsize':14,'verticalalignment':'top'})
    plt.ylabel('Accuracy',{'fontsize':14})
    plt.title('Accuracy by relations and different strategies',{'fontsize':14})
    plt.xticks(index +4* bar_width / 2, rel, rotation='90', size='14')
    plt.ylim([50,100])
    plt.legend(prop={'size': 12})
    plt.grid(True)
    plt.tight_layout()
    plt.show()
 
def draw_bar_chart_multiple_acc(data):
    index =  np.arange(len(data))
    bar_width = 0.6
    x = [item[0] for item in data]
    acc = [item[1] for item in data]
    fig, ax = plt.subplots()
    plt.bar(index+ 0.1,acc, bar_width , color=color2[3],edgecolor = "none")
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(index + bar_width / 2 + 0.1, x)
    plt.show()