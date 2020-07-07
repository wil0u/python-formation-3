# Calcul des portraits robots en fonction de la variable cible et 
# calcul du t-test pour savoir si le groupe cible est significativement different de la population totale
    
# Variables en entree :
# data = la base de donnees
# y = le nom de la variable cible
# X = la liste des noms des variables explicatives
    
# Portraits robots
import pandas as pd
import scipy.stats as sp
def portraits_robot_quanti(data, y, X):
    
 
    moy_tous = data[X].mean()
    moy_tous = pd.DataFrame(data[X].mean())
    moy_tous.rename(columns = {0:'All'}, inplace=True)
    
    moy_class = data.groupby([y], as_index = False)[X].mean()
    moy_class[y] = data[y].value_counts()
    moy_class.rename(columns= {y : 'Effectif'}, inplace=True)
    moy_class = moy_class.T

    typo = pd.concat([moy_class, moy_tous], axis=1)
    typo.loc['Effectif', 'All'] = data.shape[0]
    typo.rename(columns = dict([(i, 'y_'+ str(i)) for i in typo.columns.values]), inplace=True)
    
# Pvalues
    t_test = pd.DataFrame(columns = data[y].unique().tolist() + ['VARIABLE']) 
    for i in data[y].unique().tolist() :
        for j in range(0, len(X)) :  
            t_test.loc[j,'VARIABLE'] = X[j]
            t_test.loc[j,i] = sp.ttest_ind(data[data[y] == i][X[j]], data[data[y] != i][X[j]])[1]   
    t_test.rename(columns = dict([(i, 'p-value_'+str(i)) for i in data[y].unique().tolist()]), inplace=True)
    
# Jointure
    res = pd.merge(typo, t_test, left_index = True, right_on = 'VARIABLE', how='outer')
    res = res[['VARIABLE'] + res.columns.sort_values()[1:].tolist()]
    res = res.sort_values(by = 'VARIABLE')
    res = pd.concat([res[res['VARIABLE']=='Effectif'], res[res['VARIABLE']!='Effectif']],axis=0)
    res=res.reset_index()
    return res