"""
This script is used to run the optimization-based bit selection technique for the RTN-based PUF
"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optimization_algorithm.OptimizationAlgorithm import Comparison, Evaluation, read_config, RandomPairsGen, ParejasEval, ParentSelection, CrossOver, Mutation, SurvivorSelection
from evaluation.EvalOpt import Eval, Average_HW, ParejasEval_HW
import configparser

file_path_1 = 'data\\Taula 1 - IDVD LIN BACKWARD +5V.xlsx'
file_path_2 = 'data\\Taula 2 - IDVD SAT BACKWARD +5V.xlsx'
file_path_3 = 'data\\Taula 3 - IDVG SAT IV1 FORWARD -20V.xlsx'
file_path_4 = 'data\\Taula 4 - IDVG LIN IV1 FORWARD -18V.xlsx'
file_path_array = [file_path_1,file_path_2,file_path_3,file_path_4]
sheet_name = 'Hoja1'
effects = ['aging','thermal','electrical','all']

for data in [1,2,3,4]:
    file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\' + file_path_array[data-1]
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    data_currents_aging = df[['T0', 'T1']].to_numpy()
    data_currents_thermal = df[['T1', 'T2']].to_numpy()
    data_currents_electrical = df[['T2', 'T3']].to_numpy()
    data_currents_all = df[['T0','T1','T2','T3']].to_numpy()

    n_meas = 2
    n_meas_all = 4
    n_ttos = data_currents_aging.shape[0]
    comp_offset = 0
    n_pairs = 37

    # Optimization Algorithm parameters
    os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\optimization_algorithm')
    config = read_config()
    n_gen = config.getint('parameters', 'n_gen')
    mutation_rate = config.getfloat('parameters', 'mutation_rate')
    pob_size = config.getint('parameters', 'pob_size')
    n_offspring = config.getint('parameters', 'n_offspring')
    n_runs = config.getint('parameters', 'n_runs')
    stopping_crit = config.getboolean('parameters', 'stopping_crit')
    stop_limit = config.getint('parameters', 'stop_limit')
    fitness = config.get('parameters', 'fitness')
    P = config.getfloat('parameters', 'P')
    
    n_runs = 25
    pob_size = 100
        
    comp_aging,dif_aging,dic_parejas = Comparison(data_currents_aging,n_meas,n_ttos,comp_offset)
    parejas_eval_aging,sorted_parejas_eval = Evaluation(comp_aging,n_meas,n_ttos)

    comp_thermal,dif_thermal,dic_parejas = Comparison(data_currents_thermal,n_meas,n_ttos,comp_offset)
    parejas_eval_thermal,sorted_parejas_eval = Evaluation(comp_thermal,n_meas,n_ttos)

    comp_electrical,dif_electrical,dic_parejas = Comparison(data_currents_electrical,n_meas,n_ttos,comp_offset)
    parejas_eval_electrical,sorted_parejas_eval = Evaluation(comp_electrical,n_meas,n_ttos)

    comp_all,dif_all,dic_parejas = Comparison(data_currents_all,n_meas_all,n_ttos,comp_offset)
    parejas_eval_all,sorted_parejas_eval = Evaluation(comp_all,n_meas_all,n_ttos)

    effects = ['aging','thermal','electrical','all']
    for effect in effects:
        if effect == 'all':
            parejas_eval = parejas_eval_all
            n_meas = n_meas_all
        elif effect == 'thermal':
            parejas_eval = parejas_eval_thermal
        elif effect == 'electrical':
            parejas_eval = parejas_eval_electrical
        elif effect == 'aging':
            parejas_eval = parejas_eval_aging
            
        os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF')
        for run in range(n_runs):
                inicio = time.time()
                count = 0

                # Initialization of a variable to save algorithm data
                data_GA = np.zeros((n_gen,2))

                # Generación de población inicial
                population = RandomPairsGen(pob_size,n_pairs,n_ttos)
                # file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\' + 'data\\Combinations 75 devices - case 1.xlsx'
                # df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                # population_data = df[['Device A', 'Device B']].to_numpy()
                # population = np.tile(population_data[0:37, :].reshape(1, 74), (pob_size, 1)) + 1
                
                # Save the initial population as a file
                population_file =  'optimization_algorithm//no_opt_results//data_'+ str(data) + '//' + effect +'//no_opt_population_'+ fitness + '_run_' + str(run)
                np.savetxt(population_file, population, delimiter=",",fmt='%d')

                # Evaluation of the initial population
                fitness_population,NSP,Rel,Rel_all,prob,HW = ParejasEval(population,parejas_eval,n_ttos,n_meas, P, fitness,dic_parejas)
                print('The initial HW of the population is ' + str(np.nanmean(HW)))
                print('The initial Rel of the population is ' + str(np.nanmean(fitness_population)))
                # Optimization loop
                for gen in range(n_gen):
                    
                    # Parent Selection
                    parents,fitness_parents = ParentSelection(fitness_population,population,n_ttos,n_meas,n_offspring, rank=True)
                    
                    # Offspring generation
                    crossoved_children = CrossOver(parents,n_ttos,n_offspring)
                    offspring = Mutation(crossoved_children,mutation_rate)

                    # Offspring evaluation
                    fitness_offspring,NSP,Rel,Rel_all,prob,HW = ParejasEval(offspring,parejas_eval,n_ttos,n_meas, P, fitness, dic_parejas)
                    
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
                        if (count == stop_limit) or (new_ave_fit == 1.0):
                            break

                fin = time.time()
                print("The run of the optimization took " +  str(fin - inicio) + " seconds")
                
                # Save optimized population as a file
                population_file =  'optimization_algorithm//opt_results//data_'+ str(data) + '//' + effect +'//opt_population_'+ fitness + '_run_' + str(run)
                np.savetxt(population_file, population, delimiter=",",fmt='%d')

        # Evaluation of the optimized populations
        optimization_array = ['no_opt','opt']
        for optimization in optimization_array:
            lista = np.zeros((n_ttos,n_runs+1))
            lista[:,0] = range(1,n_ttos+1)
            population_from_lista = np.zeros((n_runs,n_ttos))
            for run in range(n_runs):
                os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\optimization_algorithm\\')
                population_file =   optimization + '_results//data_' + str(data) + '//' + effect +'\\'+ optimization +'_population_'+ fitness + '_run_' + str(run)
                population = np.genfromtxt(population_file, delimiter=',')
                
                NSP,Rel,Rel_all,prob,HW,GR = ParejasEval_HW(population, parejas_eval, n_meas, P, dic_parejas)
                NSP_mean,NSP_max,NSP_min,NSP_std,Rel_mean,Rel_std,Rel_all,HW_mean,HDinter = Average_HW(NSP, Rel, Rel_all,prob, HW, GR)
                data_1 = np.array([NSP_mean, NSP_std, Rel_all, HW_mean[0], HDinter[0]])
            
                os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\evaluation\\')
                output_file ='data_' + str(data) + '//' + effect +'\\Rel_'+ optimization +'_'+ fitness +'_run_' + str(run)
                np.savetxt(output_file, data_1, delimiter=",")
                
                for j in range(n_pairs):
                    ttor_1 = int(population[0,int(2*j)])
                    ttor_2 = int(population[0,int(2*j+1)])
                    lista[ttor_1-1,run+1],lista[ttor_2-1,run+1]  = int(ttor_2),int(ttor_1)

                #population_from_lista = 
            os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\optimization_algorithm\\pair_selection\\')
            data_file =  'data_' + str(data) +'//' + effect +'\\pair_selection_'+ optimization
            np.savetxt(data_file,lista,delimiter=",",fmt='%d')
            data_1 = np.loadtxt(data_file, delimiter=",").astype(int)
            num_columns = data_1.shape[1]
            header = "Tto & " + " & ".join([f"Run {i+1}" for i in range(num_columns - 1)])
            latex_table = "\\begin{table}\n  \\centering\n  \\scriptsize\n  \\caption{Optimized pairs for " + str(data) + " and " + effect + ".}\n  \\label{tab_pairs}\n\\begin{tabular}{c " + "c " * (num_columns - 1) + "}\n\\hline\n" + header + " \\\\\n\\hline\n"
            for row in data_1:
                latex_table += " & ".join(map(str, row)) + " \\\\\n"
            latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
            table_file = 'data_' + str(data) +'//' + effect +'\\pair_selection_'+ optimization + '.tex'
            with open(table_file, 'w') as f:
                f.write(latex_table)
                
            num_unique_numbers_per_row = np.zeros(lista.shape[0])
            for i in range(lista.shape[0]):
                unique_numbers = np.unique(lista[i,1:])
                num_unique_numbers_per_row[i] = len(unique_numbers)/n_runs
            num_unique_pairs = np.mean(num_unique_numbers_per_row)
            
            jaccard_index = {}
            for j in range(1,lista.shape[1]):
                pairs_1 = lista[:,j]
                jaccard_index[j] = {}
                for k in range(j+1,lista.shape[1]):
                    pairs_2 = lista[:,k]                   
                    intersection = np.sum(pairs_1 == pairs_2)
                    union = len(pairs_1)
                    jaccard_index[j][k] = intersection/union
            jaccard_values = [value for subdict in jaccard_index.values() for value in subdict.values()]
            jaccard_index_mean = np.mean(jaccard_values)
            jaccard_index_std = np.std(jaccard_values)
            np.savetxt('data_' + str(data) +'//' + effect +'\\jaccard_index_'+ optimization, np.array([jaccard_index_mean, jaccard_index_std]), delimiter=",")
            
            os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\evaluation\\')
            data_file =  'data_' + str(data) +'//' + effect +'\\Rel_'+ optimization +'_' + fitness
            for run in range(n_runs):
                data_run = data_file + '_run_' + str(run)
                data_1 = np.loadtxt(data_run,delimiter=",")
                if run == 0:
                    average_data = data_1
                else:
                    average_data = average_data + data_1
            average_data = average_data/n_runs
            np.savetxt(data_file,average_data,delimiter=",")

# Create a summary table for Rel_all values
os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\evaluation\\')
summary_data = np.zeros((len(effects), len(file_path_array)))
summary_data_std = np.zeros((len(effects), len(file_path_array)))
for i, effect in enumerate(effects):
    for j in range(len(file_path_array)):
        data_file = 'data_' + str(j+1) + '//' + effect + '\\Rel_opt_' + 'NSP'
        data_1 = np.loadtxt(data_file, delimiter=",")
        summary_data[i, j] = data_1[2]  # Assuming Rel_all is the third value in data_1
        summary_data_std[i, j] = data_1[1]  # Assuming NSP_std is the second value in data_1

# Generate LaTeX table for summary data
header = "Effect & " + " & ".join([f"Tabla {i+1}" for i in range(len(file_path_array))])
latex_table = "\\begin{table}\n  \\centering\n  \n  \\caption{Reliability values after the optimization (average of ten runs).}\n  \\label{tab:rel_optimization}\n\\begin{tabular}{c " + "c " * len(file_path_array) + "}\n\\hline\n" + header + " \\\\\n\\hline\n"
for i, effect in enumerate(effects):
    latex_table += effect + " & " + " & ".join([f"{summary_data[i, j]:.2f} $\\pm$ {summary_data_std[i, j]:.2f}" for j in range(len(file_path_array))]) + " \\\\\n"
latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
table_file = 'summary_rel_all.tex'
with open(table_file, 'w') as f:
    f.write(latex_table)

# Create a summary table for Rel_all values
os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\evaluation\\')
summary_data = np.zeros((len(effects), len(file_path_array)))
summary_data_std = np.zeros((len(effects), len(file_path_array)))
for i, effect in enumerate(effects):
    for j in range(len(file_path_array)):
        data_file = 'data_' + str(j+1) + '//' + effect + '\\Rel_no_opt_' + 'NSP'
        data_1 = np.loadtxt(data_file, delimiter=",")
        summary_data[i, j] = data_1[2]  # Assuming Rel_all is the third value in data_1
        summary_data_std[i, j] = data_1[1]  # Assuming NSP_std is the second value in data_1

# Generate LaTeX table for summary data
header = "Effect & " + " & ".join([f"Tabla {i+1}" for i in range(len(file_path_array))])
latex_table = "\\begin{table}\n  \\centering\n  \n  \\caption{Reliability values before the optimization (average of 1000 combinations).}\n  \\label{tab:rel_no_optimization}\n\\begin{tabular}{c " + "c " * len(file_path_array) + "}\n\\hline\n" + header + " \\\\\n\\hline\n"
for i, effect in enumerate(effects):
    latex_table += effect + " & " + " & ".join([f"{summary_data[i, j]:.2f} $\\pm$ {summary_data_std[i, j]:.2f}" for j in range(len(file_path_array))]) + " \\\\\n"
latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
table_file = 'summary_rel_all_no_opt.tex'
with open(table_file, 'w') as f:
    f.write(latex_table)
       
print('hello')