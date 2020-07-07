# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:46:25 2018

@author: dgr
"""
import numpy as np
def standardisation(x):     
    x-=np.mean(x) # the -= means can be read as x = x- np.mean(x)
    x/=np.std(x)
    return x

# test
# La variable __name__ varie selon le module dans lequel on se trouve durant l'exécution du programme. 
# Dans le module principal, sa valeur sera égale à __main__.
# alors que si le pgm est appellé par import  alors elle ne vaut pas main et ainsi l'instruction suivante
# ne sera pas effectué 

if __name__=="__main__":
    
    print("La loi uniforme tire un nombre entre une borne min et max selon une loi uniforme")
    a=int(input("Saisir une valeur de borne inf pour la loi uniforme : "))
    b=int(input("Saisir une valeur de borne sup pour la loi uniforme : "))
    n=int(input("Saisir le nombre de samples pour la loi uniforme : "))
    # variable aléa de loi uniforme entre 0 et 1
    tab=np.random.uniform(a,b,n)
    print("la moyenne de votre variable uniforme :{}".format(tab.mean()))
    
    print("Après Standardisation:")
    standardisation(tab)
    print("la moyenne est elle bien nulle ?{}".format(round(tab.mean())))
    
    
