# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:51:54 2016

@author: antan
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import os
import math
np.random.seed(42)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.sandbox.regression.gmm import IV2SLS, IVGMM, DistQuantilesGMM, spec_hausman

#On se donne ensuite un ensemble de quantiles associés:
import statsmodels



#Computation des CN(alpha) pour un quantile t.


#On considère en entrée des vecteurs (Z_i),i=1..n
#de dimension k (les tirages associés aux instruments)
#Regroupés dans une matrice Z 
#On tire pour cela un certain nombre de fois (J) des quantiles uniformes U_i,j, j=1...J
#associés à chacun des Z_i.


#On compute L_n,j facilement, on obtient le quantile d'ordre alpha pour un grand nombre de J


#On tire d'abord la matrice des U pour chacun des Z:

def quantiles_Uniform (Z,J):
    """
    parameters:
        Z: numpy array, all the variables
        J: integer, number of simulations
    return: 
        matrix of size n*J, where a column is one simulation 
    """
    n = Z.shape[0]
    return(np.random.uniform(size=(n, J)))
    
def variance_Matrix(Z, t):
    """
    parameters:
        Z: numpy array, all the variables
        t: float, searched quantil 
    return:
        Matrix W of weight in the calcul of L_n
    """
    x = np.shape(Z)
    result = np.sum([np.dot(Z[i, :].reshape((-1, 1)), Z[i, :].reshape((1, -1))) for i in range(x[0])], axis=0)
    return(1. / (t * (1 - t)) * np.linalg.inv(1. / x[0] * result))
    


def intermediar_Sumj(Z, U, t, j):
    """
    parameters:
        Z: numpy array, all variables
        t: float, searched quantil
        j: integer, index of the simulation
    return:
        np array, intermediate sum useful in the computation of the matrix L_n
    """
    B = (U < t).astype(int)
    x = np.shape(Z)
    result = np.sum((t - B[:, j]).reshape((-1, 1)) * Z, axis=0)
    return((1. / math.sqrt(x[0]) * result))

def Ln_Matrix (Z, t, J):
    """
    parameters:
        Z: np array, all variables
        t: float, searched quantil
        J: integer, number of simulations
    return:
        np array, estimator of theta for each simulation
    """#On peut donc avoir la matrice des L_n(j) avec la fonction suivante
    U = quantiles_Uniform(Z, J)
    Wn = variance_Matrix(Z, t)
    Ln = np.zeros(J)
    for j in range(J):
        H = intermediar_Sumj(Z, U, t, j).reshape((-1, 1))
        Ln[j] = np.dot(H.T, np.dot(Wn, H)) / 2
    return(Ln)
   
def C_nAlpha(Z, t, alpha, J):
    """
    parameters:
        Z: np array, all variables
        t: float, searched quantil
        J: integer, number of simulations
        alpha: Value for the confidence interval
    return:
        float: born of the critical region for theta
    """
    L_n = Ln_Matrix(Z, t, J)
    return np.percentile(L_n, alpha*100) 
#On a donc bien ici les quantiles suffisants et nécessaires de notre loi L_n.
#L'inverser n'est cependant pas du tout évident, et il faut maintenant
#Trouver les alpha dans l'intervalle de confiance.
    
    
#########################################################################################
#Il s'agit ici d'implémenter l'algorithme ordinaire pour trouver les coefficients associés aux quantiles


#Random Walk MCMC Algorithm: (Pour trouver les inverses)



#Etape 1: estimation du nouveau potentiel état pour la chaîne de Markov
def Step1 (Theta, Thetat, B):
    return(Thetat + np.random.multivariate_normal(Theta-Thetat,B))
    

#Moment généralisé
def L_n (theta, t, Y, Z):
    """
    parameters:
        theta: np array, parameters of the quantil regression
        t: float, searched quantil
        Y: np array, Explicated variables
        Z: np array, all variables
    return:
        float: 
    """
    Wn = variance_Matrix(Z, t)
    x = Z.shape
    temp = np.dot(Z, theta)
    temp = t - (Y < temp).astype(int)
    H = np.sum(temp.reshape((-1, 1)) * Z, axis=0)
    H = (1. / np.sqrt(x[0])) * H
    U = H.dot(Wn.dot(H.T))
    return U


#Etape 2 de l'algorithme: changement d'état pour la chaîne de Markov associée
def Step2(thetat,thetaprop,t,Y,Z):
    u=np.random.uniform()
    try:
        Born = min(1, np.exp(-(L_n(thetaprop, t, Y, Z)-L_n(thetat, t, Y, Z))))
    except:
        print("Too High L_n")
        Born=1
    if u<Born:
        return(thetaprop)
    else:
        return(thetat)

                
#Permet de faire l'algo total comprenant l'ensemble des étapes détaillées par le papier (étape 1, 2 et 3).        
def BoucleFinale(ThetaInit,ThetaRef,t, alpha, Y, Z, B, J):
    C_n = C_nAlpha(Z, t, alpha, J)
    U = ThetaInit.copy()
    T = ThetaInit.copy()
    ThetaProp = ThetaInit.copy()
    ThetaT = ThetaInit.copy()
    for j in range(J):
        ThetaProp=Step1(ThetaRef,ThetaT,B)
        ThetaT=Step2(ThetaT,ThetaProp,t,Y,Z)
        if L_n(ThetaProp,t,Y,Z) < C_n:
            T = np.minimum(T, ThetaProp)
            U = np.maximum(U, ThetaProp)
        if j%100==0:
            print("Iteration",j) #Comme l'algo est lent, je mets ici un "compteur" de boucles pour se rendre compte de l'avancement.
    return(T, U) #Renvoie la grosse matrice cochonne




#Estime les intervalles de confiance non asymptotiques à partir de la matrice cochonne très simplement.
def Non_Asymptotic_CF (ThetaInit,ThetaRef,t,alpha,Y,Z,B,J):
    print("Boucle Finale et Constitution de la Matrice:")
    T, U = BoucleFinale(ThetaInit, ThetaRef, t, alpha, Y, Z, B, J)
    return(np.hstack((T.reshape((-1, 1)), U.reshape(-1, 1))))
        
    
#à Partir d'un theta initial, d'un theta de référence (issu d'une régression quantile)
#On peut obtenir un intervalle de confiance non asymptotique.

#Dans la partie suivante, nous obtiendrons donc en sortie les coefficients de la régression quantile.

########################################################################
############################################################################"




#############################################################################################################
#############################################################################################################

#Le code suivant permet de générer les paramètres, les intervalles de confiance désirés, et de les comparer avec les intervalles de
#Confiance non asymptotique.


def Final_Regression (Y,Z,quantile,alpha,J): #Permet de faire la régression finale POUR 1 quantile.
    """
    parameters:
        Y: np array, Explicated variables
        Z: np array, all variables
        quantile: float, quantile of regression
        alpha : percentage of Confidence interval (usually 0.95)
    return:
        ModelTot: DataFrame, with the results of quantile regression, Asymptotic and non asymp CI
    """
    mod = statsmodels.regression.quantile_regression.QuantReg(Y,Z)
    res = mod.fit(q=quantile)
    Asymptotic_CI = res.conf_int()
    ThetaRef = res.params
    #I=statsmodels.regression.linear_model.OLS(H[0],H[1]).hessian(ThetaRef)
    I = np.diag((Asymptotic_CI[:, 1] - Asymptotic_CI[:, 0])/100)
    x = Z.shape
    Non_Asymptotic_CI = Non_Asymptotic_CF(ThetaRef, ThetaRef, quantile, alpha, Y, Z, I, J)  
    ModelTot = np.hstack((ThetaRef.reshape((-1,1)),Non_Asymptotic_CI,Asymptotic_CI))
    ModelTot = pd.DataFrame(ModelTot, columns=['Estimate', 'Non Asymp lb', 'Non Asymp ub', 'Asymp lb', 'Asymp ub'])
    return(ModelTot)       

#np.shape(Final_Regression(Y,Z,0.1,1000))

    
def hyperpyramid_generator (n,d):
    L = np.random.uniform(low=0,high=1,size=(n,d))
    Quantiles_Vec = np.random.uniform(size=n)
    Ones = np.ones(d)
    Y = Quantiles_Vec * np.dot(L, Ones)
    return(Y, L)

if __name__ == "__main__":

    H = hyperpyramid_generator(10000, 2)
    Test1 = Final_Regression(H[0],H[1],0.50,0.95,5000)
    print(Test1)
