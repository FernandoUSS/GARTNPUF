"""
In this script the optimization-based bit selection technique for the RTN-based PUF is coded
"""

import numpy as np
from scipy.special import comb
import time
import os
import random
import configparser

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
            MCF[tran,meas] = I_max - I_min
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
    parejas_eval = np.zeros((total_n_pairs,2))
    for i in range(total_n_pairs):
        num_ones = np.count_nonzero(data_Comp[i,:])
        if num_ones > n_meas/2:
            p = num_ones/n_meas
        elif num_ones < n_meas/2:
            p = (n_meas-num_ones)/n_meas
        else:
            p = 0.5
        parejas_eval[i,:] = [i,p]
    sorted_index = np.argsort(parejas_eval[:, 1])
    sorted_index = sorted_index[::-1]
    sorted_parejas_eval = parejas_eval[sorted_index]
    return parejas_eval,sorted_parejas_eval

def RandomPairsGen(n_imp,n_pairs,n_ttos):
    """ Function to generate random pairs """
    parejas = np.zeros((n_imp,2*n_pairs))
    for imp in range(n_imp):
        ttos = list(range(1,n_ttos+1))
        for pair in range(n_pairs):
            tto_choice = random.choice(ttos)
            parejas[imp,2*pair] = tto_choice
            index = ttos.index(tto_choice)
            del ttos[index]
            tto_choice = random.choice(ttos)
            parejas[imp,2*pair+1] = tto_choice
            index = ttos.index(tto_choice)
            del ttos[index]
    return parejas

def ParejasEval(parejas,parejas_eval,n_meas,P,fitness):
    """ Function to obtain the number of stable pairs (NSP) vs the probability P_0 """
    n_imp = parejas.shape[0]
    n_pairs = int(parejas.shape[1]/2)
    if P == False:
        prob = np.linspace(0.5,1,int(n_meas/2))
    else:
        prob = np.array([P])
    NSP = np.zeros((n_imp,len(prob)))
    Rel = np.zeros((n_imp,len(prob)))
    Rel_all = np.zeros(n_imp)
    pairs_eval = np.zeros((n_pairs,2))
    lista = np.concatenate((np.array(range(1,int(2*n_pairs))),[0]))
    aux = np.cumsum(lista[::-1])
    for imp in range(n_imp):
        parejas_imp = parejas[imp,:]
        for pair in range(n_pairs):
            pareja = parejas_imp[[int(2*pair),int(2*pair +1)]]
            if pareja[0]>pareja[1]:
                pareja[0],pareja[1] = pareja[1],pareja[0]
            index_pair = aux[int(pareja[0]-1)] + (pareja[1]-pareja[0]) - 1
            pairs_eval[pair,:] = parejas_eval[int(index_pair),:]
        for k in range(len(prob)):
            P_0 = prob[k]
            stable_pairs = pairs_eval[:, 1] >= P_0
            NSP[imp,k] = np.sum(stable_pairs)
            Rel[imp,k] = np.mean(pairs_eval[stable_pairs,1])
        Rel_all[imp] = np.mean(pairs_eval[:,1])
    if P != False:
        NSP = NSP.reshape(n_imp,)
    if fitness == 'NSP':
        fitness =  NSP
    elif fitness == 'Rel_all':
        fitness =  Rel_all
    else:
        raise ValueError("Not a valid fitness function")
    return fitness,NSP,Rel,Rel_all,prob

def Average(NSP,Rel,P_mean):
    NSP_mean = np.mean(NSP,axis = 0)
    Rel_mean = np.mean(Rel,axis = 0)
    P_mean = np.mean(P_mean)
    NSP_max = np.max(NSP,axis = 0)
    NSP_min = np.min(NSP,axis = 0)
    NSP_std = np.std(NSP,axis = 0)
    Rel_std = np.std(Rel,axis = 0)
    return NSP_mean,NSP_max,NSP_min,NSP_std,Rel_mean,Rel_std,P_mean

def ParentSelection(metric_parejas,parejas,n_meas,n_par,rank):
    n_comb = parejas.shape[0]
    n_ttos = parejas.shape[1]
    lista = np.zeros((n_ttos,n_comb+1))
    lista[:,0] = range(1,n_ttos+1)
    for k in range(n_comb):
        p = parejas[k,:]
        for j in range(int(n_ttos/2)):
            ttor_1 = int(p[int(2*j)])
            ttor_2 = int(p[int(2*j+1)])
            lista[ttor_1-1,k+1],lista[ttor_2-1,k+1]  = ttor_2,ttor_1
    transposed = np.transpose(lista[:,1:])
    _, unique_index = np.unique(transposed, axis=0, return_index=True)
    unique_index = np.sort(unique_index)
    metric_unique_sol = metric_parejas[unique_index].ravel()
    if rank == True:
        ind_ordenados = np.argsort(metric_unique_sol)
        ind_max_unique = ind_ordenados[-300:]
        pos = np.array(list(range(len(ind_max_unique))))
        prob_sel = 1 - np.exp(-pos/n_comb)
        normalized_metric = prob_sel/np.sum(prob_sel)
        cdf_metric = np.cumsum(normalized_metric)
        ind_par_unique = np.empty(n_par)
        for par in range(n_par):
            ind_par_1 = np.searchsorted(cdf_metric, np.random.rand())
            ind_par_unique[par] = ind_max_unique[ind_par_1]
        ind_par_unique_int = ind_par_unique.astype(np.int32)
        metric_parents = metric_unique_sol[ind_par_unique_int]
        ind_par = unique_index[ind_par_unique_int]
        parents = parejas[ind_par,:]
    else:
        normalized_metric = metric_unique_sol/np.sum(metric_unique_sol)
        cdf_metric = np.cumsum(normalized_metric)
        ind_par_unique = np.zeros(n_par)
        for par in range(n_par):
            ind_par_unique[par] = np.searchsorted(cdf_metric, np.random.rand())
        ind_par_unique_int = ind_par_unique.astype(np.int32)
        ind_par = unique_index[ind_par_unique_int]
        metric_parents = metric_unique_sol[ind_par_unique_int]
        parents = parejas[ind_par,:]
    return parents,metric_parents

def SurvivorSelection(metric_parejas,parejas,n_meas,n_imp):
    n_comb = parejas.shape[0]
    n_ttos = parejas.shape[1]
    lista = np.zeros((n_ttos,n_comb+1))
    lista[:,0] = range(1,n_ttos+1)
    for k in range(n_comb):
        p = parejas[k,:]
        for j in range(int(n_ttos/2)):
            ttor_1 = int(p[int(2*j)])
            ttor_2 = int(p[int(2*j+1)])
            lista[ttor_1-1,k+1],lista[ttor_2-1,k+1]  = ttor_2,ttor_1
    transposed = np.transpose(lista[:,1:])
    _, unique_index = np.unique(transposed, axis=0, return_index=True)
    unique_index = np.sort(unique_index)
    metric_unique_sol = metric_parejas[unique_index].ravel()
    ind_ordenados = np.argsort(metric_unique_sol)
    ind_par_unique = ind_ordenados[-n_imp:]
    ind_par_unique_int = ind_par_unique.astype(np.int32)
    metric_parejas = metric_unique_sol[ind_par_unique_int]
    ind_par = unique_index[ind_par_unique_int]
    parejas = parejas[ind_par,:]
    return parejas,metric_parejas

def CrossOver(parents,n_children):
    n_par = parents.shape[0]
    n_ttos = parents.shape[1]
    lista = np.zeros((n_ttos,n_par+1))
    lista[:,0] = range(1,n_ttos+1)
    crossoved_children = np.zeros((n_children,n_ttos))
    for k in range(n_par):
        p = parents[k,:]
        for j in range(int(n_ttos/2)):
            ttor_1 = int(p[int(2*j)])
            ttor_2 = int(p[int(2*j+1)])
            lista[ttor_1-1,k+1],lista[ttor_2-1,k+1]  = ttor_2,ttor_1
    for l in range(int(n_children/2)):
        lista_1 = lista[:,[0,int(2*l+1),int(2*l+2)]]
        children = []
        for tto in range(n_ttos):
            if lista_1[tto,0] != 0:
                if np.any(lista_1[tto,1:] != 0):
                    tto_pair = np.random.choice(lista_1[tto,1:][lista_1[tto,1:] != 0]).astype(np.int32)
                    lista_1[tto,:] = 0
                else:
                    lista_1[tto,:] = 0
                    lista_1_flat = lista_1.flatten()
                    non_zero_elem = lista_1_flat[lista_1_flat != 0]
                    tto_pair = np.random.choice(non_zero_elem).astype(np.int32)
                lista_1[tto_pair-1,:] = 0
                index =  np.isin(lista_1, [tto+1,tto_pair])
                lista_1[index] = 0
                children.extend([tto+1,tto_pair])
            else:
                lista_1[tto,:] = 0
        crossoved_children[l,:] = np.array(children)
    return crossoved_children

def Mutation(offspring,mutation_rate):
    n_children = offspring.shape[0]
    n_ttos = offspring.shape[1]
    mutated_children = offspring
    for l in range(n_children):
        tto1 = np.random.randint(1, n_ttos)
        tto2 = np.random.randint(1, n_ttos)
        if  np.random.rand() < mutation_rate:
            mutated_children[l,[tto1-1,tto2-1]] = mutated_children[l,[tto2-1,tto1-1]]
    return mutated_children

def read_config(filename='config.ini'):
    config = configparser.ConfigParser()
    config.read(filename)
    return config

# Load of Parameters of the algorithm
config = read_config()
n_gen = config.get('settings', 'n_gen')
mutation_rate = config.get('settings', 'mutation_rate')
pob_size = config.getint('settings', 'pob_size')
n_offspring = config.get('settings', 'n_offspring')
n_runs = config.get('settings', 'n_runs')
t_MCF_20 = config.getint('settings', 't_MCF_20')
t_meas_20 = config.get('settings', 't_meas_20')
T = config.get('settings', 'T')
n_ttos = config.getint('settings', 'n_ttos')
n_pairs = config.get('settings', 'n_pairs')
n_meas_T = config.get('settings', 'n_meas_T')
n_meas = config.getint('settings', 'n_meas')
comp_offset = config.getint('settings', 'comp_offset')
t_MCF_adp = config.get('settings', 't_MCF_adp')
stopping_crit = config.get('settings', 'stopping_crit')
stop_limit = config.getint('settings', 'stop_limit')
k_b = config.get('settings', 'k_b')
Ea_adp = config.get('settings', 'Ea_adp')
fitness = config.get('settings', 'fitness')
P = config.get('settings', 'P')

def main():
    os.chdir('C:\\Users\\Usuario\\Desktop\\Bit Selection')
    
    # Calculation of the MCF
    MCF_allT = np.empty((n_ttos,0))
    for i in range(len(T)):
        input_file_T = 'data/data_' + str(T[i]) +'.txt'
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

    # Several runs of the optimization algorithm
    for run in range(n_runs):
        inicio = time.time()
        count = 0

        # Initialization of a variable to save algorithm data
        data_GA = np.zeros((n_gen,2))

        # Generación de población inicial
        population = RandomPairsGen(pob_size,n_pairs,n_ttos)

        # Evaluation of the initial population
        fitness_population,NSP_parejas,Rel,P_mean_parejas,P_mean = ParejasEval(population,parejas_eval,n_meas, P, fitness) # Evaluation of the initial population
        
        # Optimization loop
        for gen in range(n_gen):
            
            # Parent Selection
            parents,fitness_parents = ParentSelection(fitness_population,population,n_meas,n_offspring, rank=True)
            
            # Offspring generation
            crossoved_children = CrossOver(parents,n_offspring)
            offspring = Mutation(crossoved_children,mutation_rate)

            # Offspring evaluation
            fitness_offspring,NSP,Rel,P_mean_offspring,P_mean = ParejasEval(offspring,parejas_eval,n_meas, P, fitness)

            # Survivor Selection
            fitness_all = np.concatenate((fitness_offspring,fitness_population),axis=0)                     # Fitness function for population + offspring
            population_all = np.concatenate((offspring,population),axis=0)                                  # Population + Offspring
            population,fitness_population = SurvivorSelection(fitness_all,population_all,n_meas,pob_size)   # Survivor selection

            # Write data_GA and console display
            new_ave_fit= np.mean(fitness_population)
            data_GA[gen,:] = np.array([gen,new_ave_fit])
            print('In the generation ' + str(gen) + ' the average fitness function of the population is ' + str(new_ave_fit))

            # Stopping criteria
            if stopping_crit == True:
                if gen != 0:
                    if new_ave_fit == data_GA[gen-1,1]:
                        count = count + 1
                if count == stop_limit:
                    break

        fin = time.time()
        print("The run of the optimization took " +  str(fin - inicio) + " seconds")

if __name__ == "__main__":
    main()