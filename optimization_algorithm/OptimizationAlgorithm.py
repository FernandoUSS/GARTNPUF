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
            MCF[tran,meas] = float(I_max) - float(I_min)
    return MCF

def Comparison(MCF,n_meas,n_ttos,comp_offset):
    """ This function evaluate the comparison of the transistor MCF """
    dic_parejas = {}
    total_n_pairs = int(comb(n_ttos,2))
    i = 0
    data_Comp = np.full((total_n_pairs,n_meas),np.nan)
    dif_MCF_pair = np.full((total_n_pairs,n_meas),np.nan)
    for j in range(1, n_ttos + 1):
        for k in range(j + 1, n_ttos + 1):
            dic_parejas[i] = [j,k]
            MCF_pair = MCF[[j-1,k-1],:]
            dif_MCF_pair[i,:]  = MCF_pair[1,:] - MCF_pair[0,:]
            #data_Comp[i,:] = np.copy(dif_MCF_pair[i,:])
            i = i + 1
    data_Comp[dif_MCF_pair < -comp_offset] = 0        
    data_Comp[dif_MCF_pair > comp_offset] = 1
    data_Comp[(data_Comp != 1) & (data_Comp != 0)] = np.random.choice([0, 1], size=np.count_nonzero((data_Comp != 1) & (data_Comp != 0)))
    return data_Comp,dif_MCF_pair,dic_parejas

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

def ParejasEval(parejas,parejas_eval,n_ttos,n_meas,P,fitness):
    """ Function to obtain the number of stable pairs (NSP) vs the probability P_0 """
    n_imp = parejas.shape[0]
    n_pairs = int(parejas.shape[1]/2)
    if P == 'all':
        prob = np.linspace(0.5,1,int(n_meas/2))
    else:
        prob = np.array([P])
    NSP = np.zeros((n_imp,len(prob)))
    Rel = np.zeros((n_imp,len(prob)))
    HW = np.zeros((n_imp,len(prob)))
    Rel_all = np.zeros(n_imp)
    pairs_eval = np.zeros((n_pairs,3))
    lista = np.concatenate((np.array(range(1,int(n_ttos))),[0]))
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
            pairs_eval[pair,[0,1]] = parejas_eval[int(index_pair),[0,1]]
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
        Rel_all[imp] = np.mean(pairs_eval[:,1])
    if P != 'all':
        NSP = NSP.reshape(n_imp,)
    if fitness == 'NSP':
        fitness =  NSP
    elif fitness == 'Rel_all':
        fitness =  Rel_all
    else:
        raise ValueError("Not a valid fitness function")
    return fitness,NSP,Rel,Rel_all,prob,HW

def ParejasEval_v2(MCF,population,n_ttos,n_meas,P,fitness):
    n_pairs = int(population.shape[1]/2)
    n_imp = int(population.shape[0])
    GR = np.full((n_imp,n_pairs),np.nan)
    p_value = np.zeros((n_imp,n_pairs))
    if P == 'all':
        prob = np.linspace(0.5,1,int(n_meas/2))
    else:
        prob = np.array([P])
    NSP = np.zeros((n_imp,len(prob)))
    Rel = np.zeros((n_imp,len(prob)))
    HW = np.zeros((n_imp,len(prob)))
    Rel_all = np.zeros(n_imp)
    for imp in range(n_imp):
        data_Comp = np.empty((n_pairs,n_meas))
        for p in range(n_pairs):
            pair = (population[imp,[int(2*p),int(2*p+1)]] - 1).astype(int)
            for meas in range(n_meas):
                MCF_pair = MCF[pair,meas]
                dif_MCF = MCF_pair[1] - MCF_pair[0]
                if dif_MCF > 0:
                    data_Comp[p,meas] = 1
                else:
                    data_Comp[p,meas] = 0
            num_ones = np.sum(data_Comp[p,:])
            if num_ones > int(n_meas/2):
                p_value[imp,p] = int(num_ones)/n_meas
                GR[imp,p] = 1
            elif num_ones < int(n_meas/2):
                p_value[imp,p] = int(n_meas - num_ones)/n_meas
                GR[imp,p] = 0
            else:
                p_value[imp,p] = 0.5
                GR[imp,p] = 0.5
        for k in range(len(prob)):
            P_0 = prob[k]
            stable_pairs = p_value[imp, :] >= P_0
            if p_value[imp, :][stable_pairs].size > 0:
                NSP[imp,k] = np.sum(stable_pairs)
                Rel[imp,k] = np.mean(p_value[imp, :][stable_pairs])
                HW[imp,k]  = np.mean(GR[imp, :][stable_pairs])
            else:
                NSP[imp,k] = 0
                Rel[imp,k] = np.nan
                HW [imp,k] = np.nan
        Rel_all[imp] = np.mean(p_value[imp,:])
    if P != 'all':
        NSP = NSP.reshape(n_imp,)
    if fitness == 'NSP':
        fitness =  NSP
    elif fitness == 'Rel_all':
        fitness =  Rel_all
    else:
        raise ValueError("Not a valid fitness function")    
    return fitness,NSP,Rel,Rel_all,prob,HW,p_value,data_Comp,GR

def ParentSelection(metric_parejas,parejas,n_ttos,n_meas,n_offspring,rank):
    """ Function to select the parent solutions """
    n_comb = parejas.shape[0]
    n_pairs = int(parejas.shape[1]/2)
    lista = np.zeros((n_ttos,n_comb+1))
    lista[:,0] = range(1,n_ttos+1)
    n_par = int(2*n_offspring)
    for k in range(n_comb):
        p = parejas[k,:]
        for j in range(n_pairs):
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

def SurvivorSelection(metric_parejas,parejas,n_ttos,n_meas,n_imp):
    """ Function to select the survivor for the next generation """
    n_comb = parejas.shape[0]
    n_pairs = int(parejas.shape[1]/2)
    lista = np.zeros((n_ttos,n_comb+1))
    lista[:,0] = range(1,n_ttos+1)
    for k in range(n_comb):
        p = parejas[k,:]
        for j in range(n_pairs):
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

def CrossOver(parents,n_ttos,n_children):
    """ Function to carry out the crossover """
    n_par = parents.shape[0]
    n_2pairs = parents.shape[1]
    lista = np.zeros((n_ttos,n_par+1))
    lista[:,0] = range(1,n_ttos+1)
    crossoved_children = np.zeros((n_children,n_2pairs))
    for k in range(n_par):
        p = parents[k,:]
        for j in range(int(n_2pairs/2)):
            ttor_1 = int(p[int(2*j)])
            ttor_2 = int(p[int(2*j+1)])
            lista[ttor_1-1,k+1],lista[ttor_2-1,k+1]  = ttor_2,ttor_1
    for l in range(n_children):
        lista_1 = lista[:,[0,int(2*l+1),int(2*l+2)]]
        children = []
        #i = 0
        #for tto in range(n_ttos):
        while len(children) != n_2pairs:
            non_zero_tto = lista_1[:,0][lista_1[:,0] != 0]
            tto = np.random.choice(non_zero_tto)
            #if lista_1[tto,0] != 0:
            if np.any(lista_1[int(tto-1),1:] != 0):
                tto_pair = np.random.choice(lista_1[int(tto-1),1:][lista_1[int(tto-1),1:] != 0]).astype(np.int32)
                lista_1[int(tto-1),:] = 0
            else:
                lista_1[int(tto-1),:] = 0
                lista_1_flat = lista_1.flatten()
                non_zero_elem = lista_1_flat[lista_1_flat != 0]
                tto_pair = np.random.choice(non_zero_elem).astype(np.int32)
            lista_1[tto_pair-1,:] = 0
            index =  np.isin(lista_1, [tto,tto_pair])
            lista_1[index] = 0
            children.extend([tto,tto_pair])
                #i = i + 1
                #if i == int(n_2pairs/2):
                #    break
            #else:
                #lista_1[tto,:] = 0
        crossoved_children[l,:] = np.array(children)
    return crossoved_children

def Mutation(offspring,mutation_rate):
    """ Function to carry out the mutation """
    n_children = offspring.shape[0]
    n_ttos = offspring.shape[1]
    mutated_children = offspring
    for l in range(n_children):
        tto1 = np.random.randint(1, n_ttos)
        tto2 = np.random.randint(1, n_ttos)
        if  np.random.rand() < mutation_rate:
            mutated_children[l,[tto1-1,tto2-1]] = mutated_children[l,[tto2-1,tto1-1]]
    return mutated_children

def read_config(filename='configGA.ini'):
    """ Function to read the config file """
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def OptAlg():
    os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\optimization_algorithm')

    # Load of Parameters of the algorithm
    config = read_config()
    data_type = config.get('parameters','data_type')
    n_gen = config.getint('parameters', 'n_gen')
    mutation_rate = config.getfloat('parameters', 'mutation_rate')
    pob_size = config.getint('parameters', 'pob_size')
    n_offspring = config.getint('parameters', 'n_offspring')
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
    stopping_crit = config.getboolean('parameters', 'stopping_crit')
    stop_limit = config.getint('parameters', 'stop_limit')
    fitness = config.get('parameters', 'fitness')
    P = config.getfloat('parameters', 'P')
    k_b = config.getfloat('constants', 'k_b')
    Ea_adp = config.getfloat('constants', 'Ea_adp')
    n_meas = n_meas_T*len(T)

    os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF')

    # Calculation of the MCF
    MCF_allT = np.empty((n_ttos,0))
    for i in range(len(T)):
        if data_type == 1:
            input_file_T = 'data/data_set_1/data_' + str(T[i]) +'.txt'
        elif data_type == 2:
            input_file_T = 'data/data_set_2/data_2_' + str(T[i])
        else:
            raise ValueError("Select an appropiate data_type: exp or sim")
        if t_MCF_adp == True:
            t_MCF = t_MCF_20*np.exp(Ea_adp/k_b*(1/T[i] - 1/T[2]))
            t_meas = t_meas_20*np.exp(Ea_adp/k_b*(1/T[i] - 1/T[2]))
        else:
            t_MCF = t_MCF_20
            t_meas = t_meas_20
        MCF_T = CalculoMCF(input_file_T,t_MCF,t_meas,n_meas_T,n_ttos)
        MCF_allT = np.concatenate((MCF_allT,MCF_T),axis = 1)
    #np.savetxt('MCF_allT',MCF_allT,delimiter = ',')
    #MCF_allT = np.loadtxt('MCF_allT',delimiter = ',')
    
    # Comparison and fitness evaluation of all possible pairs
    data_Comp,dif_MCF_pair,dic_parejas = Comparison(MCF_allT,n_meas,n_ttos,comp_offset)
    #np.savetxt('data_Comp', data_Comp, delimiter = ',')
    #data_Comp = np.loadtxt('data_Comp', delimiter = ',')
    parejas_eval,sorted_parejas_eval = Evaluation(data_Comp,n_meas,n_ttos)
    #np.savetxt('parejas_eval', parejas_eval, delimiter = ',')
    #parejas_eval = np.loadtxt('parejas_eval', delimiter = ',')
    
    # Several runs of the optimization algorithm
    for run in range(n_runs):
        inicio = time.time()
        count = 0

        # Initialization of a variable to save algorithm data
        data_GA = np.zeros((n_gen,2))

        # Generación de población inicial
        population = RandomPairsGen(pob_size,n_pairs,n_ttos)

        # Save the initial population as a file
        if data_type == 1:
            population_file =  'optimization_algorithm//no_opt_results//experimental//initial_population_'+ fitness + '_run_' + str(run)
        elif data_type == 2:
            population_file =  'optimization_algorithm//no_opt_results//simulated//initial_population_'+ fitness + '_run_' + str(run)
        else:
            raise ValueError("Select an appropiate data_type: 1 or 2")
        np.savetxt(population_file, population, delimiter=",")

        # Evaluation of the initial population
        fitness_population,NSP,Rel,Rel_all,prob,HW = ParejasEval(population,parejas_eval,n_ttos,n_meas, P, fitness)
        print('Thi initial HW of the population is ' + str(np.nanmean(HW)))
        
        # Optimization loop
        for gen in range(n_gen):
            
            # Parent Selection
            parents,fitness_parents = ParentSelection(fitness_population,population,n_ttos,n_meas,n_offspring, rank=True)
            
            # Offspring generation
            crossoved_children = CrossOver(parents,n_ttos,n_offspring)
            offspring = Mutation(crossoved_children,mutation_rate)

            # Offspring evaluation
            #fitness_offspring,NSP,Rel,P_mean_offspring,P_mean = ParejasEval_v2(MCF_allT,offspring,n_ttos,n_meas,P,fitness)
            fitness_offspring,NSP,Rel,Rel_all,prob,HW = ParejasEval(offspring,parejas_eval,n_ttos,n_meas, P, fitness)
            
            # Survivor Selection
            fitness_all = np.concatenate((fitness_offspring,fitness_population),axis=0)                     # Fitness function for population + offspring
            population_all = np.concatenate((offspring,population),axis=0)                                  # Population + Offspring
            population,fitness_population = SurvivorSelection(fitness_all,population_all,n_ttos,n_meas,pob_size)   # Survivor selection

            # Write data_GA and console display
            new_ave_fit= np.mean(fitness_population)
            data_GA[gen,:] = np.array([gen,new_ave_fit])
            print('In the generation ' + str(gen) + ' the average fitness function of the population is ' + str(new_ave_fit))
            print('In the generation ' + str(gen) + ' the average HW of the population is ' + str(np.nanmean(HW)))

            # Stopping criteria
            if stopping_crit == True:
                if gen != 0:
                    if new_ave_fit == data_GA[gen-1,1]:
                        count = count + 1
                    else:
                        count = 0
                if count == stop_limit:
                    break

        fin = time.time()
        print("The run of the optimization took " +  str(fin - inicio) + " seconds")
        
        # Save optimized population as a file
        if data_type == 1:
            population_file =  'optimization_algorithm//opt_results//experimental//population_optimized_'+ fitness + '_run_' + str(run)
        elif data_type == 2:
            population_file =  'optimization_algorithm//opt_results//simulated//population_optimized_'+ fitness + '_run_' + str(run)
        else:
            raise ValueError("Select an appropiate data_type: 1 or 2")
        np.savetxt(population_file, population, delimiter=",")

#if __name__ == "__main__":
#    main()