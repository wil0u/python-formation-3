# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:59:14 2018

@author: dgr
"""

import numpy as np
# calcul le lift tx de rep dans la classe / tx de reponse global sur le prmier quantile de score (proba décroissante)
# si on met 10, cela renvoie le premier decile
# si on met 5 cela renvoie le premier vingtile
def lift(proba,X,reponse,p=10):
    #p = 10
    sorted_proba = np.array(list(reversed(np.argsort(proba))))
    positives = sum(reponse)
    tp = sum(np.array(reponse)[sorted_proba[:int(round((p*X.shape[0])/100,0))]])
    lift = round(100*tp/(float(positives)*p),2)
    print("lift at " '{} percent : {}'.format(p,lift))
    return lift

from sklearn.metrics import  roc_curve , auc
import matplotlib.pyplot as plt
def auc_et_roc(Y,proba):
    fpr, tpr, thresholds = roc_curve(Y,proba)
    roc_auc = auc(fpr, tpr)
    print("auc=",roc_auc)
    plt.clf()
    plt.plot(fpr, tpr, label='Courbe ROC (Aire = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
import pandas as pd
def CAP_table(pred, true, stepsize, n):
    
    # Pour chaque décile de score on calcule la valeur du Lift

    # pred = score  
    # true = vraie valeur de y
    # stepsize = écart entre les quantiles
    # n = combien de quantiles
    # par exemple : n=4 et stepsize=5 signifie chaque 5% jusqu'à 20%)

    DF = pd.concat([true, pred], axis = 1).reset_index(drop=True)
    DF.columns = [ 'T', 'Prob']
    DF = DF.sort_values('Prob', ascending = False)
    DF.index = np.arange(1, len(DF)+1)
    truey = DF['T'].sum()
    All = len(DF)
    df = pd.DataFrame()
    P_W_A = truey / DF.shape[0]
    
    for i in range(1, n+1):
        
        p_upp = (i*stepsize)/100.0
        p_low = ((i-1)*stepsize)/100.0
        q_upp = np.int(round(All*p_upp,0))
        q_low = np.int(round(All*p_low,0))
        
        # take the part from q_low to q_upp
        data = DF.iloc[q_low:q_upp]
        A = len(data)
        W = data['T'].sum()
        P = round((W/A)*100,2)
        data_cum = DF.iloc[0:q_upp]
        A_cum = len(data_cum)
        W_cum = data_cum['T'].sum()
        P_cum = round((W_cum/truey)*100,2)
        W_R = round((W_cum/A_cum)*100,2)
        L_cum = round(W_R/(P_W_A*100),2)
        
        df = pd.concat([df, pd.DataFrame({'Population': [A], 
                                          'Positifs parmi la population': [W], 
                                          '% positifs': [P],
                                          'Population cumulée': [A_cum], 
                                          'Positifs cumulés': [W_cum], 
                                          '% positifs cumulés sur le total des positifs': [P_cum],
                                          'Lift' : [L_cum], 
                                          '% Positifs cumulés' : [W_R]}, 
                                         index = [i * stepsize])])
        
    Data = df[['Population', 'Positifs parmi la population', '% positifs',
               'Population cumulée', 'Positifs cumulés', '% Positifs cumulés',
               '% positifs cumulés sur le total des positifs', 'Lift']]
    
    return(Data)


import numpy as np
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes,                       
                          title=None,
                          cmap=plt.cm.Blues):
   
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Variable cible observé',
           xlabel='Variable cible prédite')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax