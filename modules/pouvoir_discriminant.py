import pandas as pd
def pouvoir_discriminant(dataframe, y, X):
    
    # Calcul le pouvoir discriminant des variables entre les modalites X et la cible y
    
    # En entree : 
    # le dataframe (ex: 'train')
    # le nom de la variable cible y
    # la liste des noms des variables explicatives
    
    X.append(y)
    data = dataframe[X] 
    
    n = data.shape[0]
    p = data.shape[1]-1
    KK =  len(data[y].value_counts())
    
    Moy= {}
    VW = {}
    VB = {}
    VT = {}
    effectifs = pd.DataFrame(data[y].value_counts())
    
    #For each variable 
    for i in range(0,p):
        
        #Global Var, moy and for each cluster
        VT[data.columns[i]] = n*data[data.columns[i]].var(ddof = 0)
        Moy[data.columns[i]] = data[data.columns[i]].mean()
        Moyc = pd.DataFrame(data.groupby([y])[data.columns[i]].mean())
        Varc = pd.DataFrame(data.groupby([y])[data.columns[i]].var(ddof = 0))

        #Var within
        var_temp = pd.merge(Varc, effectifs, left_index = True, right_index = True, how = 'inner')
        VW[data.columns[i]] = pd.DataFrame(var_temp[var_temp.columns[0]]* var_temp[var_temp.columns[1]]).sum()[0]

        #Var Between
        moy_temp = pd.merge(Moyc, effectifs, left_index = True, right_index = True, how = 'inner')
        moy_temp['Temp'] = moy_temp[moy_temp.columns[0]].apply(lambda x: (x-Moy[data.columns[i]])**2)
        VB[data.columns[i]] = pd.DataFrame(moy_temp[moy_temp.columns[1]]*moy_temp['Temp']).sum()[0]
    
    VB = pd.DataFrame(VB.items(), columns = ['Variable', 'VB'])
    VW = pd.DataFrame(VW.items(), columns = ['Variable', 'VW'])
    VT = pd.DataFrame(VT.items(), columns = ['Variable', 'VT'])
    
    Pdiscri = pd.merge(VB, VW, left_on = 'Variable', right_on = 'Variable', how= 'inner')
    Pdiscri = pd.merge(Pdiscri, VT, left_on = 'Variable', right_on = 'Variable', how= 'inner')
    
    Pdiscri['Calinski'] = (Pdiscri['VB']/(KK-1))/(Pdiscri['VW']/(n-KK))
    Pdiscri['TxVarWithin'] = ((Pdiscri['VW']/Pdiscri['VT'])*100).round(2)
    Pdiscri['TxVarBetween'] = ((Pdiscri['VB']/Pdiscri['VT'])*100).round(2)
    
    X.remove(y)
    
    return Pdiscri.sort_values(['TxVarBetween'], ascending = [False])