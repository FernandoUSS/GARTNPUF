"""
In this script the functions to evaluate the optimization results are coded
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import time
import os
import math
import random
from scipy.stats import multivariate_normal
import matplotlib.patches as patches
import matplotlib
import configparser
import warnings

def CalculoMCF(input_file,t_MCF,t_meas,n_meas,n_ttos):
    """ This function evaluates the MCF of the transistors """
    data = np.genfromtxt(input_file, delimiter=',')
    n_points = data.shape[1]
    n_points_MCF = np.floor(t_MCF/t_meas).astype(int) 
    n_meas_1 =  np.floor(n_points/n_points_MCF).astype(int) 
    if n_meas > n_meas_1:
        n_meas = n_meas_1
    MCF = np.zeros((n_ttos,n_meas))
    for tran in range(n_ttos):
        data_tran = data[tran,:]
        for meas in range(n_meas):
            ini = np.floor(meas*n_points_MCF).astype(int) 
            fin = np.floor((meas+1)*n_points_MCF).astype(int) 
            inter = list(range(ini, fin))
            for ind in inter:
                I = data_tran[ind]
                if ind == ini:
                    I_min = I
                    I_max = I
                if I > I_max:
                    I_max = I
                elif I < I_min:
                    I_min = I 
            MCF[tran,meas] = float(I_max) - float(I_min)
    return MCF

def Comparison(MCF,n_meas,n_ttos,comp_offset):
    """ This function evaluate the comparison of the transistor MCF """
    dic_parejas = {}
    total_n_pairs = int(comb(n_ttos,2))
    i = 0
    data_Comp = np.zeros((total_n_pairs,n_meas))
    for j in range(1, n_ttos + 1):
        for k in range(j + 1, n_ttos + 1):
            dic_parejas[i] = [j,k]
            MCF_pair = MCF[[j-1,k-1],:]
            dif_MCF_pair = MCF_pair[0,:] - MCF_pair[1,:]
            data_Comp[i,:] = np.copy(dif_MCF_pair)
            i = i + 1
    data_Comp[data_Comp > comp_offset] = 1
    data_Comp[data_Comp < -comp_offset] = 0
    data_Comp[(data_Comp != 1) & (data_Comp != 0)] = np.random.choice([0, 1], size=np.count_nonzero((data_Comp != 1) & (data_Comp != 0)))
    return data_Comp,dic_parejas

def Evaluation(data_Comp,n_meas,n_ttos):
    """ Evaluation of the stability in the pairs response """
    total_n_pairs = int(comb(n_ttos,2))
    parejas_eval = np.zeros((total_n_pairs,3))
    for i in range(total_n_pairs):
        num_ones = np.count_nonzero(data_Comp[i,:])
        if num_ones > n_meas/2:
            p = num_ones/n_meas
            GR = 1
        elif num_ones < n_meas/2:
            p = (n_meas-num_ones)/n_meas
            GR = 0
        else:
            p = 0.5
            GR = 0.5
        parejas_eval[i,:] = [i,p,GR]
    sorted_index = np.argsort(parejas_eval[:, 1])
    sorted_index = sorted_index[::-1]
    sorted_parejas_eval = parejas_eval[sorted_index]
    return parejas_eval,sorted_parejas_eval

def ParejasEval_HW(parejas,parejas_eval,n_meas,P):
    """ Function to obtain the number of stable pairs (NSP) vs the probability P_0 """
    n_imp = parejas.shape[0]
    n_pairs = int(parejas.shape[1]/2)
    if P == 'all':
        prob = np.linspace(0.5,1,int(n_meas/2))
    else:
        prob = np.array([P])
    NSP = np.zeros((n_imp,len(prob)))
    Rel = np.zeros((n_imp,len(prob)))
    HW = np.empty((n_imp,len(prob)))
    P_mean = np.zeros(n_imp)
    GR = np.zeros((n_imp,n_pairs))
    pairs_eval = np.zeros((n_pairs,3))
    lista = np.concatenate((np.array(range(1,int(2*n_pairs))),[0]))
    aux = np.cumsum(lista[::-1])
    for imp in range(n_imp):
        parejas_imp = parejas[imp,:]
        for pair in range(n_pairs):
            pareja = parejas_imp[[int(2*pair),int(2*pair +1)]]
            GR_change = False
            if pareja[0]>pareja[1]:
                pareja[0],pareja[1] = pareja[1],pareja[0]
                GR_change = True
            index_pair = aux[int(pareja[0]-1)] + (pareja[1]-pareja[0]) - 1
            pairs_eval[pair,:] = parejas_eval[int(index_pair),:]
            if GR_change == True:
                pairs_eval[pair,2] = 1 - parejas_eval[int(index_pair),2]
            else:
                pairs_eval[pair,2] =  parejas_eval[int(index_pair),2]
        for k in range(len(prob)):
            P_0 = prob[k]
            stable_pairs = pairs_eval[:, 1] >= P_0
            if pairs_eval[stable_pairs,1].size > 0:
                NSP[imp,k] = np.sum(stable_pairs)/n_pairs
                Rel[imp,k] = np.mean(pairs_eval[stable_pairs,1])
                HW[imp,k]  = np.mean(pairs_eval[stable_pairs,2]) 
            else:
                NSP[imp,k] = 0
                Rel[imp,k] = np.nan
                HW [imp,k] = np.nan
        GR[imp,:] = pairs_eval[:,2]
        P_mean[imp] = np.mean(pairs_eval[:,1])
    if P != 'all':
        NSP = NSP.reshape(n_imp,)
    return NSP,Rel,P_mean,prob,HW,GR

def Average_HW(NSP,Rel,P_mean,prob,HW,GR):
    n_imp = NSP.shape[0]
    n_pairs = GR.shape[1]
    NSP_mean = np.mean(NSP,axis = 0)
    Rel_mean = np.nanmean(Rel,axis = 0)
    P_mean =  np.mean(P_mean)
    HW_mean = np.nanmean(HW,axis = 0)
    NSP_max = np.max(NSP,axis = 0)
    HW_max =  np.max(HW,axis = 0)
    NSP_min = np.min(NSP,axis = 0)
    HW_min =  np.min(HW,axis = 0)
    NSP_std = np.std(NSP,axis = 0)
    Rel_std = np.nanstd(Rel,axis = 0)
    HW_std =  np.nanstd(HW,axis = 0)
    # Evaluation HDinter
    k = n_imp
    total_distance = 0
    pair_count = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            total_distance += np.sum(GR[i,:] != GR[j,:])
            pair_count += n_pairs
    HDinter = total_distance/pair_count
    HDinter =  HDinter*np.ones(len(prob),)
    return NSP_mean,NSP_max,NSP_min,NSP_std,Rel_mean,Rel_std,P_mean,HW_mean,HDinter

def AverageRun(n_runs,data_type,opt,fitness):
    if (data_type == 1) & (opt == False):
        data_file =  'data_set_1//Rel_no_optimized_'+ fitness
    elif (data_type == 2) & (opt == False):
        data_file =  'data_set_2//Rel_no_optimized_'+ fitness
    elif (data_type == 1) & (opt == True):
        data_file =  'data_set_1//Rel_optimized_'+ fitness
    elif (data_type == 2) & (opt == True):
        data_file =  'data_set_2//Rel_optimized_'+ fitness  
    else:
        raise ValueError("Select an appropiate data_type: exp or sim and an appropiate opt: True or False")
    for run in range(n_runs):
        data_run = data_file + '_run_' + str(run)
        data = np.loadtxt(data_run,delimiter=",")
        if run == 0:
            average_data = data
        else:
            average_data = average_data + data
    average_data = average_data/n_runs
    np.savetxt(data_file,average_data,delimiter=",")
    return average_data

def read_config(filename='configEval.ini'):
    """ Function to read the config file """
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def Eval():
    warnings.simplefilter('error', Warning)
    os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\evaluation')

    # Load of Parameters of the algorithm
    config = read_config()
    data_type = config.getint('parameters','data_type')
    opt = config.getboolean('parameters','opt')
    n_runs = config.getint('parameters', 'n_runs')
    t_MCF_20 = config.getfloat('parameters', 't_MCF_20')
    t_meas_20 = config.getfloat('parameters', 't_meas_20')
    T_str = config.get('parameters', 'T')
    T = [int(T1) for T1 in T_str.split(',')]
    n_ttos = config.getint('parameters', 'n_ttos')
    n_pairs = config.getint('parameters', 'n_pairs')
    n_meas_T = config.getint('parameters', 'n_meas_T')
    comp_offset = config.getfloat('parameters', 'comp_offset')
    t_MCF_adp = config.getboolean('parameters', 't_MCF_adp')
    k_b = config.getfloat('constants', 'k_b')
    Ea_adp = config.getfloat('constants', 'Ea_adp')
    n_meas = n_meas_T*len(T)
    fitness = config.get('parameters', 'fitness')
    P = config.getfloat('parameters', 'P')

    os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF')

    # Calculation of the MCF
    MCF_allT = np.empty((n_ttos,0))
    for i in range(len(T)):
        if data_type == 1:
            input_file_T = 'data/data_set_1/data_' + str(T[i]) +'.txt'
        elif data_type == 2:
            input_file_T = 'data/data_set_2/data_2_' + str(T[i])
        else:
            raise ValueError("Select an appropiate data_type: 1 or 2")
        if t_MCF_adp == True:
            t_MCF = t_MCF_20*np.exp(Ea_adp/k_b*(1/T[i] - 1/T[2]))
            t_meas = t_meas_20*np.exp(Ea_adp/k_b*(1/T[i] - 1/T[2]))
        else:
            t_MCF = t_MCF_20
            t_meas = t_meas_20
        MCF_T = CalculoMCF(input_file_T,t_MCF,t_meas,n_meas_T,n_ttos)
        MCF_allT = np.concatenate((MCF_allT,MCF_T),axis = 1)
    
    # Comparison and fitness evaluation of all possible pairs
    data_Comp,dic_parejas = Comparison(MCF_allT,n_meas,n_ttos,comp_offset)
    parejas_eval,sorted_parejas_eval = Evaluation(data_Comp,n_meas,n_ttos)

    # Evaluation of the optimized populations
    for run in range(n_runs):
        os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\optimization_algorithm')
        if (data_type == 1) & (opt == False):
            population_file =  'no_opt_results//data_set_1//initial_population_'+ fitness + '_run_' + str(run)
        elif (data_type == 2) & (opt == False):
            population_file =  'no_opt_results//data_set_2//initial_population_'+ fitness + '_run_' + str(run)
        elif (data_type == 1) & (opt == True):
            population_file =  'opt_results//data_set_1//population_optimized_'+ fitness + '_run_' + str(run)
        elif (data_type == 2) & (opt == True):
            population_file =  'opt_results//data_set_2//population_optimized_'+ fitness + '_run_' + str(run)    
        else:
            raise ValueError("Select an appropiate data_type: 1 or 2 and an appropiate opt: True or False")
        population = np.genfromtxt(population_file, delimiter=',')
        NSP,Rel,P_mean,prob,HW,GR = ParejasEval_HW(population, parejas_eval, n_meas, P ='all')
        NSP_mean,NSP_max,NSP_min,NSP_std,Rel_mean,Rel_std,P_mean,HW_mean,HDinter = Average_HW(NSP, Rel, P_mean,prob, HW, GR)
        data = np.concatenate((NSP_mean[:,None],NSP_std[:,None],Rel_mean[:,None],Rel_std[:,None],HW_mean[:,None],HDinter[:,None]),axis=1)
        os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\evaluation\\')
        if (data_type == 1) & (opt == False):
            output_file = 'data_set_1//Rel_no_optimized_'+ fitness +'_run_' + str(run)
        elif (data_type == 2) & (opt == False):
            output_file = 'data_set_2//Rel_no_optimized_'+ fitness +'_run_' + str(run)
        elif (data_type == 1) & (opt == True):
            output_file =  'data_set_1//Rel_optimized_'+ fitness +'_run_' + str(run)
        elif (data_type == 2) & (opt == True):
            output_file =  'data_set_2//Rel_optimized_'+ fitness +'_run_' + str(run)       
        else:
            raise ValueError("Select an appropiate data_type: 1 or 2 and an appropiate opt: True or False")
        np.savetxt(output_file, data, delimiter=",")
    # Averaging between different runs
    average_data = AverageRun(n_runs,data_type,opt,fitness)
