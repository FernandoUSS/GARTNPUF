"""
In this script different functions to simulate charge trapping in transistors are coded
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

def DefGen(N_duts,mu,sigma,mu_l,sigma_l,mu_u,sigma_u,R,Eae,Eac,sigma_Eae,sigma_Eac,N_ave_def):
    """ Generation of transistor defects characteristics """
    N_defs = np.zeros(N_duts)
    emission_constants = {}
    capture_constants = {}
    current_shifts = {}
    activation_energies_e = {}
    activation_energies_c = {}
    for dut in range(N_duts):
        N_defs[dut] = np.random.poisson(N_ave_def)
        emission_constants[dut] = []
        capture_constants[dut] = []
        current_shifts[dut] = []
        activation_energies_e[dut] = []
        activation_energies_c[dut] = []
        for defs in range(int(N_defs[dut])):
            time_constants = np.random.multivariate_normal(mu, sigma, size=1)
            emission_constants[dut].append(10**time_constants[0,0])
            capture_constants[dut].append(10**time_constants[0,1])
            activation_energies_e[dut].append(np.random.normal(Eae, sigma_Eae, size=1)[0])
            activation_energies_c[dut].append(np.random.normal(Eac, sigma_Eac, size=1)[0])
            if np.random.rand() < R:
                current_shift = np.random.normal(mu_l, sigma_l, size=1)
                current_shifts[dut].append(10**current_shift[0])
            else:
                current_shift = np.random.normal(mu_u, sigma_u, size=1)
                current_shifts[dut].append(10**current_shift[0])
        emission_list = emission_constants[dut]
        capture_list = capture_constants[dut]      
        current_list = current_shifts[dut]
        activation_e_list = activation_energies_e[dut]
        activation_c_list = activation_energies_c[dut]
        activation_energies_e[dut] = np.array(activation_e_list)
        activation_energies_c[dut] = np.array(activation_c_list)
        emission_constants[dut] = np.array(emission_list)
        capture_constants[dut] = np.array(capture_list)
        current_shifts[dut] =  np.array(current_list)
    return emission_constants, capture_constants, current_shifts, activation_energies_e, activation_energies_c

def DefGenTZV(N_duts,mu,sigma,mu_l,sigma_l,mu_u,sigma_u,R,Eae,Eac,sigma_Eae,sigma_Eac,N_ave_def,mu_Vth,sigma_Vth):
    """ Generation of transistor defects characteristics """
    N_defs = np.zeros(N_duts)
    emission_constants = {}
    capture_constants = {}
    current_shifts = {}
    activation_energies_e = {}
    activation_energies_c = {}
    Vth = {}
    for dut in range(N_duts):
        N_defs[dut] = np.random.poisson(N_ave_def)
        emission_constants[dut] = []
        capture_constants[dut] = []
        current_shifts[dut] = []
        activation_energies_e[dut] = []
        activation_energies_c[dut] = []
        Vth[dut] = np.random.normal(mu_Vth, sigma_Vth, size=1)
        for defs in range(int(N_defs[dut])):
            time_constants = np.random.multivariate_normal(mu, sigma, size=1)
            emission_constants[dut].append(10**time_constants[0,0])
            capture_constants[dut].append(10**time_constants[0,1])
            activation_energies_e[dut].append(np.random.normal(Eae, sigma_Eae, size=1)[0])
            activation_energies_c[dut].append(np.random.normal(Eac, sigma_Eac, size=1)[0])
            if np.random.rand() < R:
                current_shift = np.random.normal(mu_l, sigma_l, size=1)
                current_shifts[dut].append(10**current_shift[0])
            else:
                current_shift = np.random.normal(mu_u, sigma_u, size=1)
                current_shifts[dut].append(10**current_shift[0])
        emission_list = emission_constants[dut]
        capture_list = capture_constants[dut]    
        current_list = current_shifts[dut]
        activation_e_list = activation_energies_e[dut]
        activation_c_list = activation_energies_c[dut]
        activation_energies_e[dut] = np.array(activation_e_list)
        activation_energies_c[dut] = np.array(activation_c_list)
        emission_constants[dut] = np.array(emission_list)
        capture_constants[dut] = np.array(capture_list)
        current_shifts[dut] =  np.array(current_list)
    return emission_constants, capture_constants, current_shifts, activation_energies_e, activation_energies_c, Vth

def DefSel(N_duts,t_min,t_max, emission_constants, capture_constants, current_shifts, activation_energies_e, activation_energies_c):
    """ Selection of the defects that can appear in the experimental window """
    selected_emission_constants = {}
    selected_capture_constants = {} 
    selected_current_shifts = {}
    selected_activation_energies_e = {}
    selected_activation_energies_c = {}
    for dut in range(N_duts):
        mask_tau_e = (emission_constants[dut] >= t_min) & (emission_constants[dut] <= t_max)
        mask_tau_c = (capture_constants[dut] >= t_min) & (capture_constants[dut] <= t_max)
        mask_both =  mask_tau_e & mask_tau_c
        selected_emission_constants[dut] = emission_constants[dut][mask_both]
        selected_capture_constants[dut] = capture_constants[dut][mask_both]
        selected_current_shifts[dut] = current_shifts[dut][mask_both]
        selected_activation_energies_e[dut] = activation_energies_e[dut][mask_both]
        selected_activation_energies_c[dut] = activation_energies_c[dut][mask_both]
    return selected_emission_constants, selected_capture_constants, selected_current_shifts, selected_activation_energies_e, selected_activation_energies_c

def InitialState(selected_emission_constants,selected_capture_constants):
    """ Sample the initial state of the defects """
    initial_states = {}
    N_duts = len(selected_emission_constants)
    for dut in range(N_duts):
        N_defs = selected_emission_constants[dut].shape[0]
        initial_states[dut] = np.zeros(N_defs)
        for deft in range(N_defs):
            if selected_emission_constants[dut][deft]/(selected_emission_constants[dut][deft] + selected_capture_constants[dut][deft]) > np.random.rand():   # the defect is occupied
                initial_states[dut][deft] = 1
            else:                                                                                                                                            # the defect is unoccupied
                initial_states[dut][deft] = 0
    return initial_states

def CompetingClocks(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,t_step,t_max):
    """ Creation of current traces based on the algorithm of competing clocks """
    n_points = int(t_max/t_step)
    times = np.linspace(0,t_max,n_points)
    transition_times = {}
    voltage_threshold = {}
    N_duts = len(initial_states)
    for dut in range(N_duts):
        t=t_step
        transition_times[dut] = [t]
        N_defs = selected_emission_constants[dut].shape[0]
        state = initial_states[dut]
        voltage_threshold[dut] = [Vth_TZV[dut] + state @ selected_current_shifts[dut] + np.random.normal(0, 2e-10, 1)]
        while t<t_max:
            sampled_times = np.zeros(N_defs)
            for defs in range(N_defs):
                if state[defs] == 1:  # the defect is occupied
                    sampled_times[defs] = np.random.exponential(scale=selected_capture_constants[dut][defs], size=1)
                else:                 # the defect is unoccupied
                    sampled_times[defs] = np.random.exponential(scale=selected_emission_constants[dut][defs], size=1)
            chosen_def = np.argmin(sampled_times)
            chosen_time = sampled_times[chosen_def]
            t = t + chosen_time
            transition_times[dut].append(t)
            state[chosen_def] = not state[chosen_def]
            voltage_threshold[dut].append(Vth_TZV[dut] + state @ selected_current_shifts[dut] + np.random.normal(0, 2e-9, 1))
    return times,voltage_threshold

def ExponentialTimes(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,t_step,t_max):
    """ Creation of current traces based on the algorithm of DTMC with exponential times """
    n_points = int(t_max/t_step)
    times = np.linspace(0,t_max,n_points)
    transition_times = {}
    voltage_threshold = {}
    N_duts = len(initial_states)
    for dut in range(N_duts):
        t=t_step
        transition_times[dut] = [t]
        state = initial_states[dut]
        voltage_threshold[dut] = [Vth_TZV[dut] + state @ selected_current_shifts[dut] + np.random.normal(0, 2e-9, 1)]
        while t<t_max:
            transitions_rate =  np.multiply(state, 1/selected_emission_constants[dut]) + np.multiply(np.logical_not(state).astype(int), 1/selected_capture_constants[dut])
            tau = 1/sum(transitions_rate)
            sampled_time = np.random.exponential(scale=tau, size=1)
            t = t + sampled_time[0]
            transition_times[dut].append(t)
            cdf = tau*np.cumsum(transitions_rate)
            chosen_def = np.searchsorted(cdf, np.random.rand())
            state[chosen_def] = not state[chosen_def]
            voltage_threshold[dut].append(Vth_TZV[dut] + state @ selected_current_shifts[dut])
    indexes = np.searchsorted(transition_times[dut], times)
    voltage_threshold[dut] = np.array(voltage_threshold[dut])[indexes] + np.random.normal(0, 2e-10, n_points)
    return times,voltage_threshold

def ExponentialTimeswithMCF(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,t_min,t_max,t_MCF):
    """ Creation of current traces based on the algorithm of DTMC with exponential times """
    times = np.linspace(t_min,t_max,10000)
    transition_times = {}
    voltage_threshold = {}
    MCFs = {}
    N_duts = len(initial_states)
    n_meas = np.floor((t_max - t_min)/t_MCF)
    t_MCF_list = t_MCF*np.arange(1,n_meas+1)
    for dut in range(N_duts):
        t=t_min
        transition_times[dut] = [t]
        state = initial_states[dut]
        voltage = state @ selected_current_shifts[dut]
        voltage_threshold[dut] = [voltage]
        MCFs[dut] = []
        i = 0
        while t<t_max:
            min_voltage = voltage
            max_voltage = voltage
            while t < t_MCF_list[i]:
                transitions_rate =  np.multiply(state, 1/selected_emission_constants[dut]) + np.multiply(np.logical_not(state).astype(int), 1/selected_capture_constants[dut])
                tau = 1/sum(transitions_rate)
                sampled_time = np.random.exponential(scale=tau, size=1)
                t = t + sampled_time[0]
                transition_times[dut].append(t)
                cdf = tau*np.cumsum(transitions_rate)
                chosen_def = np.searchsorted(cdf, np.random.rand())
                state[chosen_def] = not state[chosen_def]
                voltage = state @ selected_current_shifts[dut]
                voltage_threshold[dut].append(voltage)
                if voltage > max_voltage:
                    max_voltage = voltage
                if voltage < min_voltage:
                    min_voltage = voltage
            MCFs[dut].append(max_voltage - min_voltage)
            i = i + 1
        indexes = np.searchsorted(transition_times[dut], times)
        voltage_threshold[dut] = np.array(voltage_threshold[dut])[indexes]
    return transition_times,voltage_threshold,MCFs

def Probability(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,t_max,t_step):
    """ Emulation of sampled RTN traces """
    N_duts = len(initial_states)
    n_points = int(t_max/t_step)
    times = np.linspace(0,t_max,n_points)
    voltage_threshold = np.empty((N_duts,n_points))
    for dut in range(N_duts):
        state_dut = initial_states[dut]
        N_defs = selected_emission_constants[dut].shape[0]
        P_tran = {}
        tau_0 = np.empty(N_defs,)
        P_0 = np.empty(N_defs,)
        P_1 = np.empty(N_defs,)
        for deft in range(N_defs):
            tau_0[deft] = selected_capture_constants[dut][deft]*selected_emission_constants[dut][deft]/(selected_emission_constants[dut][deft] + selected_capture_constants[dut][deft]) # Fix this, is so slow
            P_0[deft] = selected_capture_constants[dut][deft]/(selected_capture_constants[dut][deft]+ selected_emission_constants[dut][deft])
            P_1[deft] = selected_emission_constants[dut][deft]/(selected_capture_constants[dut][deft]+ selected_emission_constants[dut][deft])
            P_tran[deft] = np.empty((2,2))
            P_tran[deft][0,0] = P_0[deft] + P_1[deft]*np.exp(-t_step/tau_0[deft])
            P_tran[deft][0,1] = P_1[deft] - P_1[deft]*np.exp(-t_step/tau_0[deft])
            P_tran[deft][1,0] = P_0[deft] - P_0[deft]*np.exp(-t_step/tau_0[deft])
            P_tran[deft][1,1] = P_1[deft] + P_0[deft]*np.exp(-t_step/tau_0[deft])
        for step in range(n_points):
            for deft in range(N_defs):
                if state_dut[deft] == 1:
                    rho = np.array([0,1])
                else:
                    rho = np.array([1,0])
                prob_tran = np.matmul(rho,P_tran[deft])
                cdf = np.cumsum(prob_tran)
                state_dut[deft] = np.searchsorted(cdf, np.random.rand())
            voltage_threshold[dut][step] = Vth_TZV[dut] + state_dut @ selected_current_shifts[dut] + np.random.normal(0, 1e-9, 1)
    return times,voltage_threshold

def ProbTime(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,gate_voltage,t_max,t_step,int_time):
    """ Emulation of non-stationary RTN traces based on base algorithm (under development)"""
    N_duts = len(initial_states)
    n_points = int(t_max/t_step)
    voltage_threshold = np.empty((N_duts,n_points))
    time_index = int(t_step/int_time)
    #volt_index = 
    for dut in range(N_duts):
        state_dut = initial_states[dut]
        N_defs = selected_emission_constants[dut].shape[0]
        P_tran = {}
        tau_0 = np.empty(N_defs,)
        P_0 = np.empty(N_defs,)
        P_1 = np.empty(N_defs,)
        for deft in range(N_defs):
            #for voltage in range(volt_index):
            tau_0[deft] = selected_capture_constants[dut][deft]*selected_emission_constants[dut][deft]/(selected_emission_constants[dut][deft] + selected_capture_constants[dut][deft]) # Fix this, is so slow
            P_0[deft] = selected_capture_constants[dut][deft]/(selected_capture_constants[dut][deft]+ selected_emission_constants[dut][deft])
            P_1[deft] = selected_emission_constants[dut][deft]/(selected_capture_constants[dut][deft]+ selected_emission_constants[dut][deft])
            P_tran[deft] = np.empty((2,2))
            P_tran[deft][0,0] = P_0[deft] + P_1[deft]*np.exp(-t_step/tau_0[deft])
            P_tran[deft][0,1] = P_1[deft] - P_1[deft]*np.exp(-t_step/tau_0[deft])
            P_tran[deft][1,0] = P_0[deft] - P_0[deft]*np.exp(-t_step/tau_0[deft])
            P_tran[deft][1,1] = P_1[deft] + P_0[deft]*np.exp(-t_step/tau_0[deft])
        for step in range(n_points):
            #for in range(time_index):
            for deft in range(N_defs):
                if state_dut[deft] == 1:
                    rho = np.array([0,1])
                else:
                    rho = np.array([1,0])
                prob_tran = np.matmul(rho,P_tran[deft])
                cdf = np.cumsum(prob_tran)
                state_dut[deft] = np.searchsorted(cdf, np.random.rand())
            voltage_threshold[dut][step] = Vth_TZV[dut] + state_dut @ selected_current_shifts[dut] + np.random.normal(0, 1e-9, 1)
    return voltage_threshold

def read_config(filename='configSim.ini'):
    """ Function to read the config file with the simulation parameters"""
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def Sim():
    os.chdir('C:\\Users\\Usuario\\Desktop\\GARTNPUF\\simulation')

    # Load of Parameters of the algorithm
    config = read_config()
    N_ave_def = config.getfloat('parameters', 'N_ave_def')
    mu_e = config.getfloat('parameters', 'mu_e')
    mu_c = config.getfloat('parameters', 'mu_c')
    sigma_e = config.getfloat('parameters', 'sigma_e')
    sigma_c = config.getfloat('parameters', 'sigma_c')
    rho = config.getfloat('parameters', 'rho')
    mu_l = config.getfloat('parameters', 'mu_l')
    mu_u = config.getfloat('parameters', 'mu_u')
    sigma_l = config.getfloat('parameters', 'sigma_l')
    sigma_u = config.getfloat('parameters', 'sigma_u')
    R = config.getfloat('parameters', 'R')
    Eae = config.getfloat('parameters', 'Eae')
    Eac = config.getfloat('parameters', 'Eac')
    sigma_Eae = config.getfloat('parameters', 'sigma_Eae')
    sigma_Eac = config.getfloat('parameters', 'sigma_Eac')
    N_duts = config.getint('parameters', 'N_duts')
    t_step_20 = config.getfloat('parameters', 't_step_20')
    t_max_20 = config.getfloat('parameters', 't_max_20')
    t_min_20 = config.getfloat('parameters', 't_min_20')
    mu_Vth = config.getfloat('parameters', 'mu_Vth')
    sigma_Vth = config.getfloat('parameters', 'sigma_Vth')
    T_str = config.get('parameters', 'T')
    T = np.array([int(T1) for T1 in T_str.split(',')])
    k_b = config.getfloat('constants', 'k_b')
    adapt_window = config.getboolean('parameters', 'adapt_window')
    alg = config.getint('parameters', 'alg')
    mu = np.array([mu_e,mu_c])
    sigma  = np.array([[sigma_e**2, rho*sigma_e*sigma_c],[rho*sigma_e*sigma_c, sigma_c**2]])
    T_K = T + 293

    # Generation of defects characteristics at 20 degrees
    emission_constants_20, capture_constants_20, current_shifts, activation_energies_e, activation_energies_c, Vth_TZV = DefGenTZV(N_duts,mu,sigma,mu_l,sigma_l,mu_u,sigma_u,R,Eae,Eac,sigma_Eae,sigma_Eac,N_ave_def,mu_Vth,sigma_Vth)

    # Different temperatures loop
    for i in range(len(T)):
        # Defect characteristics for the particular temperature
        emission_constants = {dut: emission_constants_20[dut]*np.exp(activation_energies_e[dut]/k_b*(1/T_K[i] - 1/293)) for dut in range(N_duts)}
        capture_constants = {dut: capture_constants_20[dut]*np.exp(activation_energies_c[dut]/k_b*(1/T_K[i] - 1/293)) for dut in range(N_duts)}

        # Simulator experimental window adjustment according to temperature
        if adapt_window == True:
            t_max = t_max_20*np.exp(0.7/k_b*(1/T_K[i] - 1/293))
            t_step = t_step_20*np.exp(0.7/k_b*(1/T_K[i] - 1/293))
            t_min = t_min_20*np.exp(0.7/k_b*(1/T_K[i] - 1/293))
        else:
            t_max = t_max_20
            t_step = t_step_20
            t_min = t_min_20
        
        # Discard non-active defects in the particular time window
        selected_emission_constants, selected_capture_constants, selected_current_shifts, selected_activation_energies_e, selected_activation_energies_c  = DefSel(N_duts,t_min,t_max,emission_constants, capture_constants, current_shifts, activation_energies_e, activation_energies_c)
        
        # Sample the initial state of defects
        initial_states = InitialState(selected_emission_constants,selected_capture_constants)

        # Algorithm to simulate RTN according to defects characteristics
        if alg == 1:
            times,vth = Probability(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,t_max,t_step)
        elif alg == 2:
            times,vth = CompetingClocks(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,t_min,t_max)
        elif alg == 3:
            times,vth = ExponentialTimes(initial_states,selected_emission_constants,selected_capture_constants,selected_current_shifts,Vth_TZV,t_min,t_max)
        else:
            raise ValueError("Select an appropiate algorithm: alg = 1, 2 or 3")
        
        # Save simulated data as a file
        output_file = 'data/data_simulated_' + str(T[i])
        np.savetxt(output_file, vth, delimiter=",")