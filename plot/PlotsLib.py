"""
In this script different functions to create figures are coded
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import time
import os
import scienceplots
import math
import random
from scipy.stats import multivariate_normal
import matplotlib.patches as patches

def TimeDistribution(axs,dut,interval,selected_emission_constants,selected_capture_constants):
    """ Creation of the data to plot in the time distribution """
    line_styles = ["-", "--", "-.", ":"]
    log_t = np.linspace(interval[0],interval[1],1000)
    t = 10**log_t
    N_defs = selected_emission_constants[dut].shape[0]
    f_e = np.zeros((N_defs,1000))
    f_c = np.zeros((N_defs,1000))
    bins_e = np.zeros((N_defs,50))
    bins_c = np.zeros((N_defs,50))
    hist_e = np.zeros((N_defs,50-1))
    hist_c = np.zeros((N_defs,50-1))
    bin_widths_e = np.zeros((N_defs,50-1))
    bin_widths_c = np.zeros((N_defs,50-1))
    for deft in range(N_defs):
        f_e[deft,:] = t*np.log(10)/selected_emission_constants[dut][deft]*np.exp(-t/selected_emission_constants[dut][deft])
        f_c[deft,:] = -t*np.log(10)/selected_capture_constants[dut][deft]*np.exp(-t/selected_capture_constants[dut][deft])
        t_e = np.random.exponential(scale = selected_emission_constants[dut][deft],size=1000)
        t_c = np.random.exponential(scale = selected_capture_constants[dut][deft] ,size=1000)
        bins_e[deft,:] = np.logspace(np.log10(min(t_e)), np.log10(max(t_e)), 50)
        bins_c[deft,:] = np.logspace(np.log10(min(t_c)), np.log10(max(t_c)), 50)
        hist_e[deft,:], bins_e[deft,:] = np.histogram(t_e, bins=bins_e[deft,:])
        hist_c[deft,:], bins_c[deft,:] = np.histogram(t_c, bins=bins_c[deft,:])
        bin_widths_e[deft,:] = np.diff(bins_e[deft,:])
        bin_widths_c[deft,:] = np.diff(bins_c[deft,:])
        log_bin_widths_e = np.diff(np.log10(bins_e[deft,:]))
        log_bin_widths_c = np.diff(np.log10(bins_c[deft,:]))
        total_data_points = len(t_e)
        hist_e[deft,:] = hist_e[deft,:] / (total_data_points*log_bin_widths_e)
        hist_c[deft,:] = -hist_c[deft,:] / (total_data_points*log_bin_widths_c)
        axs.bar(bins_e[deft,:][:-1], hist_e[deft,:], width=bin_widths_e[deft,:], color='skyblue', alpha=1, label='Normalized Histogram')
        axs.plot(t, f_e[deft,:],line_styles[deft % len(line_styles)], color='blue')
        axs.bar(bins_c[deft,:][:-1], hist_c[deft,:], width=bin_widths_c[deft,:], color='lightpink', alpha=1, label='Normalized Histogram')
        axs.plot(t, f_c[deft,:],line_styles[deft % len(line_styles)], color='red')
        #axs.set_xlabel('$t_{e,c}$ (s)', fontsize=14)
        #axs.set_ylabel('Transitions', fontsize=11)
        axs.set_yticklabels([])
        axs.set_xscale('log')
        axs.grid(True)
        axs.set_xlim(10**interval[0], 10**interval[1])
        axs.set_ylim(-1, 1)
        axs.text(1.2*1e-2, 0.75, 'Emission', fontsize=9, color='blue')
        axs.text(1.2*1e-2, -0.85, 'Capture', fontsize=9, color='red')
    return axs

def TZVdistribution(axs,n_points, mu_TZV, sigma_TZV):
    """ Creation of the TZV histograms """
    vth = np.random.normal(mu_TZV,sigma_TZV, size = n_points)
    axs.hist(vth)
    axs.set_yticklabels([])
    axs.set_ylabel('Counts', fontsize=14)
    axs.set_xlabel('$V_{th}$ (a.u.)', fontsize=14)
    return axs
    
def FigureRTNsignal(axs,dut,transition_times,voltage_threshold,t_start,t_end):
    """ Creation of a figure of the RTN signal """
    times = np.linspace(t_start,t_end,10000)
    indexes = np.searchsorted(transition_times[dut], times)
    voltage_threshold_dut = np.array(voltage_threshold[dut])[indexes]
    axs.plot(times,voltage_threshold_dut)
    #axs.set_xlabel('Time (s)', fontsize=14)
    #axs.set_ylabel('$\Delta V_{th}$', fontsize=14)
    axs.grid(True)
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.set_xticks([0,20,40,60,80,100])
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.set_xlim(t_start,t_end)
    return axs

def FigureRTNsignal2(axs,dut,voltage_threshold,t_start,t_end):
    """ Creation of a figure of the RTN signal """
    n_points = len(voltage_threshold[dut])
    times = np.linspace(t_start,t_end,n_points)
    axs.plot(times,voltage_threshold[dut])
    axs.set_xlabel('Time (s)', fontsize=14)
    axs.set_ylabel('$V_{th}$ (a.u.)', fontsize=14)
    #axs.grid(True)
    axs.set_facecolor('white')
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.set_xticks([0,20,40,60,80,100])
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    axs.set_xlim(t_start,t_end)
    return axs

def FigureMCF(axs,dut,transition_times,voltage_threshold,t_start,t_end,t_MCF):
    times = np.linspace(t_start,t_end,10000)
    indexes = np.searchsorted(transition_times[dut], times)
    voltage_threshold_dut = np.array(voltage_threshold[dut])[indexes] + np.random.normal(0, 2e-9, (len(times),1))
    low_enveloppe = np.array([])
    upper_enveloppe = np.array([])
    MCF = np.array([])
    points_MCF = t_MCF/(t_end - t_start)*10000
    for point in range(len(times)):
        if (point+1) % points_MCF == 0 or MCF.size == 0:
            low_enveloppe = np.append(low_enveloppe,voltage_threshold_dut[point])
            upper_enveloppe = np.append(upper_enveloppe,voltage_threshold_dut[point])
            MCF =  np.append(MCF,upper_enveloppe[point] - low_enveloppe[point])
            axs.plot([times[point],times[point]],[-5,1e7*upper_enveloppe[point]],color='black',linestyle = '--',linewidth=1)
            if (point+1) % points_MCF == 0:               
               axs.annotate('', 
             xy=(times[point-1], upper_enveloppe[point-1]*1e7), 
             xytext=(times[point-1], low_enveloppe[point-1]*1e7), 
             arrowprops=dict(arrowstyle='|-|, head_width=2, head_length=4',  # Custom sharp arrowheads
                             lw=5,  # Increased line width
                             color='green', 
                             linewidth=15))  # Increased arrow width
                #axs.annotate('MCF',xy=(times[point-1], low_enveloppe[point-1]*1e7 - 0.05e7*(low_enveloppe[point-1] + upper_enveloppe[point-1])),xycoords='data', ha='center', va='center', color='green',fontsize=10,fontweight='bold')
        else:
            if voltage_threshold_dut[point] < low_enveloppe[point-1]:
                low_enveloppe = np.append(low_enveloppe,voltage_threshold_dut[point])
            else:
                low_enveloppe = np.append(low_enveloppe,low_enveloppe[point-1])
            if voltage_threshold_dut[point] > upper_enveloppe[point-1]:
                upper_enveloppe = np.append(upper_enveloppe,voltage_threshold_dut[point])
            else:
                upper_enveloppe = np.append(upper_enveloppe,upper_enveloppe[point-1])
            MCF =  np.append(MCF,upper_enveloppe[point] - low_enveloppe[point])
    axs.plot(times,voltage_threshold_dut*1e7,color='black',linestyle = '-',linewidth=1,zorder=0,alpha=0.4)
    axs.plot(times,low_enveloppe*1e7,color='skyblue',linestyle = '-',linewidth=2,zorder=1,alpha = 0.7)
    axs.plot(times,upper_enveloppe*1e7,color='red',linestyle = '-',linewidth=2,zorder=2,alpha = 0.7)
    x_min = times[0]
    x_max = times[-1]
    y_min = 1e7*min(low_enveloppe)
    y_max = 1e7*max(upper_enveloppe)
    axs.annotate('', xy=(x_min,y_min - 0.1*(y_max - y_min)),xytext=(x_min+ t_MCF, y_min - 0.1*(y_max - y_min)),arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
    axs.annotate('$1^{st}$ $t_{MVF}$',xy=(0.5*(2*x_min + t_MCF),y_min - 0.2*(y_max - y_min)),xycoords='data', ha='center', va='center', color='black',fontsize=11)
    axs.annotate('$2^{nd}$ $t_{MVF}$',xy=(0.5*(2*x_min + t_MCF)+t_MCF,y_min - 0.2*(y_max - y_min)),xycoords='data', ha='center', va='center', color='black',fontsize=11)
    axs.annotate('$3^{rd}$ $t_{MVF}$',xy=(0.5*(2*x_min + t_MCF)+2*t_MCF,y_min - 0.2*(y_max - y_min)),xycoords='data', ha='center', va='center', color='black',fontsize=11)
    axs.annotate('$4^{th}$ $t_{MVF}$',xy=(0.5*(2*x_min + t_MCF)+3*t_MCF,y_min - 0.2*(y_max - y_min)),xycoords='data', ha='center', va='center', color='black',fontsize=11)
    axs.annotate('$5^{th}$ $t_{MVF}$',xy=(0.5*(2*x_min + t_MCF)+4*t_MCF,y_min - 0.2*(y_max - y_min)),xycoords='data', ha='center', va='center', color='black',fontsize=11)
    axs.set_xlabel('Time (s)', fontsize=14)
    axs.set_ylabel('$V_D$ (a.u.)', fontsize=14)
    #axs.annotate('MVF',xy=(times[int(points_MCF)], 1e7*upper_enveloppe[int(points_MCF-2)] + 0.1*(y_max - y_min)),xycoords='data', ha='center', va='center', color='green',fontsize=12,fontweight='bold')
    axs.grid(True)
    axs.set_yticklabels([])
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.set_xticklabels([])
    axs.set_xlim(x_min - 0.05*(x_max - x_min),x_max + 0.05*(x_max - x_min))
    axs.set_ylim(y_min - 0.3*(y_max - y_min),y_max + 0.3*(y_max - y_min))
    return axs

def FigureRelvsT(n_meas,simulated,t_MCF_var,opt_20,Reli,colors):
    T1 = [-20,0,20,40,60,80]
    T = [253,273,293,313,333,353]
    fig,axs = plt.subplots()
    prob = np.linspace(0.5,1,int(n_meas/2))
    prob_1 = np.linspace(0.5,1,int(6*n_meas/2))
    Reli_T = np.empty(len(T))
    for i in range(len(T)):
        if simulated == True:
            output_file = 'data/Rel_simulated_'+ str(T[i])
        elif t_MCF_var == True:
            output_file = 'data/Rel_optimized_t_MCF_T=20_'+ str(T[i])
        elif opt_20 == True:
            output_file = 'data/Rel_optimized_T=20_'+ str(T1[i])
        else:
            output_file = 'data/Rel_'+ str(T1[i])
        Rel = np.genfromtxt(output_file, delimiter=',')
        NSP = Rel[:,0]
        if Reli == True:
            Reli_T[i] = Rel[0,2]
        axs.plot(prob, NSP ,label= str(T1[i]) + '\u00B0C',color = colors[i % len(colors)]) #'\u00B0C')
    if simulated == True:
        output_file = 'data/Rel_simulated_allT'
    elif t_MCF_var == True:
        output_file = 'data/Rel_allT_optimized_t_MCF_T=20'
    elif opt_20 == True:
        output_file = 'data/Rel_allT_optimized_T=20'
    else:
        output_file = 'data/Rel_allT'
    Rel = np.genfromtxt(output_file, delimiter=',')
    NSP = Rel[:,0]
    Reli_allT = Rel[0,2]
    axs.plot(prob_1, NSP ,label= 'all T')
    axs.text(0.8, 13,'$Rel_{allT}$ = ' + str(int(Reli_allT * 1000) / 1000) ,color='blue',fontsize=10)
    axs.set_xlabel('Probability $P_0$')
    axs.set_ylabel(r'\% of Stable CRPs, $F_{P_0}$')
    axs.set_ylim(0,100)
    axs.set_xlim(0.5,1)
    axs.set_xticks([0.5,0.6,0.7,0.8,0.9,1.0])
    axs.set_xticklabels(['0.5','0.6','0.7','0.8','0.9','1.0'])
    axs.set_yticks([0,20,40,60,80,100])
    axs.set_yticklabels(['0','20','40','60','80','100'])
    axs.grid(True)
    axs.legend().set_visible(False)
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center',frameon =True, ncol=4)
    fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.1)
    if Reli == True:
        ax_inset = fig.add_axes([0.125, 0.2, 0.25, 0.25])  
        ax_inset.plot(T1,Reli_T, color='blue',marker='s',linestyle = '-', label="Rel")
        ax_inset.set_title("Rel")
        ax_inset.set_ylim(0.85,1)
        ax_inset.set_xlim(-25,85)
        ax_inset.set_xticks([-20,20,80])
        ax_inset.set_xticklabels(['-20\u00B0C', '20\u00B0C', '80\u00B0C'])
        ax_inset.set_yticks([0.9,1.0])
        ax_inset.set_yticklabels(['0.9', '1.0'])
    return axs

def FigureReloptimized(n_meas,t_MCF_var,opt_t_MCF,Reli):
    T1 = [-20,0,20,40,60,80]
    T = [253,273,293,313,333,353]
    fig,axs = plt.subplots()
    #prob = np.linspace(0.5,1,int(n_meas/2))
    prob_1 = np.linspace(0.5,1,int(6*n_meas/2))
    output_file = 'data/Rel_allT'
    Rel = np.genfromtxt(output_file, delimiter=',')
    NSP = Rel[:,0]
    Reli_allT = Rel[0,2]
    axs.text(0.8, 13,'$Rel_{allT}$ = ' + str(int(Reli_allT * 1000) / 1000) ,color='black',fontsize=9)
    axs.plot(prob_1, NSP ,label= r'Random combinations')
    output_file = 'data/Rel_allT_optimized_NSP'
    Rel = np.genfromtxt(output_file, delimiter=',')
    NSP = Rel[:,0]
    Rel_opt_NSP = Rel[0,2]
    axs.text(0.7, 65,'$Rel_{allT}$ = ' + str(int(Rel_opt_NSP * 1000) / 1000),color='red',fontsize=9)
    axs.plot(prob_1, NSP ,label= r'Optimized selection for $F_{1}$')
    output_file = 'data/Rel_allT_optimized_Rel'
    Rel = np.genfromtxt(output_file, delimiter=',')
    NSP = Rel[:,0]
    Reli_opt_Rel = Rel[0,2]
    axs.text(0.82, 82,'$Rel_{allT}$ = ' + str(int(Reli_opt_Rel * 1000) / 1000) ,color='blue',fontsize=9)
    axs.plot(prob_1, NSP ,label= r'Optimized selection for $Rel_{allT}$')
    if t_MCF_var == True:
        output_file = 'data/Rel_simulated_t_MCF_2_allT'
        Rel = np.genfromtxt(output_file, delimiter=',')
        NSP = Rel[:,0]
        axs.plot(prob_1, NSP ,label= r'Random combinations varying $t_{MCF}$')
    if opt_t_MCF == True:
        output_file = 'data/Rel_allT_optimized_P_mean_t_MCF_2'
        Rel = np.genfromtxt(output_file, delimiter=',')
        NSP = Rel[:,0]
        axs.plot(prob_1, NSP ,label= r'Optimized selection varying $t_{MCF}$')
    #axs.legend(frameon=True,fontsize = 'small')
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center',frameon =True, ncol=2)
    axs.grid(True)
    axs.set_xlabel('Probability $P_0$')
    axs.set_ylabel(r'\% of Stable CRPs, $F_{P_0}$')
    fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.1)
    axs.set_ylim(0,100)
    axs.set_xlim(0.5,1)
    axs.set_xticks([0.5,0.6,0.7,0.8,0.9,1.0])
    axs.set_xticklabels(['0.5','0.6','0.7','0.8','0.9','1.0'])
    axs.set_yticks([0,20,40,60,80,100])
    axs.set_yticklabels(['0','20','40','60','80','100'])
    Rel_data = np.empty((6,))
    Rel_opt_Rel = np.empty((6,))
    Rel_opt_NSP = np.empty((6,))
    for i in range(len(T)):
        input_file = 'data/Rel_'+ str(T1[i])
        Rel = np.genfromtxt(input_file, delimiter=',')
        Rel_data[i] = Rel[0,2]
        input_file = 'data/Rel_allT_optimized_Rel_'+ str(T1[i])
        Rel = np.genfromtxt(input_file, delimiter=',')
        Rel_opt_Rel[i] = Rel[0,2]
        input_file = 'data/Rel_allT_optimized_NSP_'+ str(T1[i])
        Rel = np.genfromtxt(input_file, delimiter=',')
        Rel_opt_NSP[i] = Rel[0,2]
    if Reli == True:
        ax_inset = fig.add_axes([0.125, 0.2, 0.25, 0.25])  
        ax_inset.plot(T1,Rel_data, color='black',marker='s',markersize = 3,linestyle = '-', label="Rel")
        ax_inset.plot(T1,Rel_opt_Rel, color='blue',marker='s',markersize =3,linestyle = '-', label="Rel")
        ax_inset.plot(T1,Rel_opt_NSP, color='red',marker='s',markersize = 3,linestyle = '-', label="Rel")
        ax_inset.set_title("Rel")
        ax_inset.set_ylim(0.85,1)
        ax_inset.set_xlim(-25,85)
        ax_inset.set_xticks([-20,20,80])
        ax_inset.set_xticklabels(['-20\u00B0C', '20\u00B0C', '80\u00B0C'])
        ax_inset.set_yticks([0.9,1.0])
        ax_inset.set_yticklabels(['0.9', '1.0'])
    return axs

def FigureRTNtemp(ax,dut,simulated,t_MCF_var):
    T = [253,273,293,313,333,353]
    T1 = [-20,0,20,40,60,80]
    for i in range(len(T)):
        if simulated == True:
            input_file = 'data/data_simulated_' + str(T[i])
        elif t_MCF_var == True:
            input_file = 'data/data_simulated_t_MCF_2_' + str(T[i])
        else:
            input_file = 'data/data_' + str(T1[i]) + '.txt'
        voltage_threshold = np.genfromtxt(input_file, delimiter=',')[dut,:]
        voltage_threshold = voltage_threshold - np.mean(voltage_threshold)
        times = np.linspace(0,100,len(voltage_threshold))
        ax[i].plot(times,voltage_threshold)
        ax[i].set_title(str(T1[i]) + '\u00B0C')
        ax[i].grid(True)
        if (i == 3) or (i == 4) or (i == 5):
            #ax[i].set_xlabel('Time (s)')
            ax[i].set_xticks([0,100])
            ax[i].set_xticklabels(['0','100 s'])
        else:
            ax[i].set_xticklabels([])
        if (i == 0) or (i == 3):
            ax[i].set_ylabel('$V_{D}$ (a.u.)')
        else:
            ax[i].set_yticklabels([])
    return ax

def FigureElipse(mean,cov,emission_constans, capture_constans,t_MCF):
    k_b = 8.617315e-5
    if t_MCF == True:
        original_figsize = plt.rcParams["figure.figsize"]
        fig, axs = plt.subplots(figsize=(original_figsize[0] * 4/3, original_figsize[1] * 1))
    else:
        original_figsize = plt.rcParams["figure.figsize"]
        fig, axs = plt.subplots(figsize=(original_figsize[0] * 1, original_figsize[1] * 5/3))
    ltau_e = np.linspace(-9, 10, 200)
    ltau_c = np.linspace(-9, 10, 200)
    X, Y = np.meshgrid(ltau_e, ltau_c)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    contour = axs.contourf(X, Y, Z, cmap='viridis', levels=30)
    axs.scatter(np.log10(emission_constans),np.log10(capture_constans),color='white',s = 6)
    if t_MCF == False:
        rect = patches.Rectangle((-6, -6), 6, 6,linewidth=2,edgecolor='skyblue',facecolor='none')  
        axs.text(-5.6, -5.5,'T = 20 \u00B0C',color='skyblue',fontsize=18)
        axs.add_patch(rect)
        rect2 = patches.Rectangle((-6 - 0.7/k_b*(1/353 - 1/293), -6 - 0.7/k_b*(1/353 - 1/293)), 6, 6,linewidth=2,edgecolor='red',facecolor='none')
        axs.text(0.2, -0.8,'T = 80 \u00B0C',color='red',fontsize=18)
        axs.add_patch(rect2)
    else:
        rect = patches.Rectangle((-6, -6), 6, 6,linewidth=2,edgecolor='skyblue',facecolor='none')  
        axs.text(-5.6, -5.5,'T = 20 \u00B0C',color='skyblue',fontsize=18)
        axs.add_patch(rect)
        rect2 = patches.Rectangle((-6 - 0.7/k_b*(1/313 - 1/293), -6 - 0.7/k_b*(1/313 - 1/293)), +6 + 0.7/k_b*(1/313 - 1/293), +6 + 0.7/k_b*(1/313 - 1/293),linewidth=2,edgecolor='orange',facecolor='none')
        axs.text(-4, -3.55,'T = 40 \u00B0C',color='orange',fontsize=18)
        axs.add_patch(rect2)
        rect3 = patches.Rectangle((-6 - 0.7/k_b*(1/273 - 1/293), -6 - 0.7/k_b*(1/273 - 1/293)), +6 + 0.7/k_b*(1/273 - 1/293), +6 + 0.7/k_b*(1/273 - 1/293),linewidth=2,edgecolor='pink',facecolor='none')
        axs.text(-7.75, -7.75,'T = 0 \u00B0C',color='pink',fontsize=18)
        axs.add_patch(rect3)
    if t_MCF == True:
        axs.set_xlim(-8.5,2)
        axs.set_ylim(-8.5,2)
        axs.set_xticks([-8,-6,-4,-2,0,2])
        axs.set_xticklabels(['-8','-6','-4','-2','0','2'])
        axs.set_yticks([-8,-6,-4,-2,0,2])
        axs.set_yticklabels(['-8','-6','-4','-2','0','2'])
        axs.tick_params(axis='x', labelsize=16)
        axs.tick_params(axis='y', labelsize=16)
    else:
        #axs.set_aspect('equal')
        axs.set_aspect('auto', adjustable='box')
        axs.set_xlim(-6.5,6)
        axs.set_ylim(-6.5,10)
        axs.set_xticks([-6,-3,0,3,6])
        axs.set_xticklabels(['-6','-3','0','3','6'])
        axs.set_yticks([-6,-3,0,3,6,9])
        axs.set_yticklabels(['-6','-3','0','3','6','9'])
        axs.tick_params(axis='x', labelsize=20, colors = 'blue')
        axs.tick_params(axis='y', labelsize=20, colors = 'blue')
        axs2 = axs.twinx()
        axs2.set_ylim(np.array(axs.get_ylim()) + 0.7/k_b*(1/353 - 1/293))
        axs3 = axs.twiny()
        axs3.set_xlim(np.array(axs.get_xlim()) + 0.7/k_b*(1/353 - 1/293))
        axs2.set_yticks([-9,-6,-3,0,3])
        axs2.set_yticklabels(['-9','-6','-3','0','3'])
        axs3.set_xticks([-9,-6,-3,0])
        axs3.set_xticklabels(['-9','-6','-3','0'])
        axs3.tick_params(axis='x', labelsize=20, colors = 'red')
        axs2.tick_params(axis='y', labelsize=20, colors = 'red')
    axs.set_xlabel(r'log($\tau_e$)', fontsize = 20)
    axs.set_ylabel(r'log($\tau_c$)', fontsize = 20)
    if t_MCF == True:
        fig.subplots_adjust(left=0.1,right=0.975,top=0.925,bottom=0.145)
    else:
        fig.subplots_adjust(left=0.125,right=0.925,top=0.925,bottom=0.1)
    if t_MCF == True:
        inset = fig.add_axes([0.135, 0.685, 0.04, 0.2])  
    else:
        inset = fig.add_axes([0.145, 0.7, 0.04, 0.2])
    colorbar = plt.colorbar(contour, cax=inset, orientation='vertical')
    colorbar.set_ticks([0, np.max(Z)])
    colorbar.set_ticklabels(['0', r'$5 \times 10^{-3}$'],fontsize = 16)
    colorbar.ax.yaxis.set_tick_params(colors='white')
    return axs

def FigureElipseEa(mean,cov):
    k_b = 8.617315e-5
    tau_0 = 55e-12
    mu = k_b*293*mean - k_b*293*np.log(tau_0)
    sigma = (k_b*293)**2*cov
    original_figsize = plt.rcParams["figure.figsize"]
    fig, axs = plt.subplots(figsize=(original_figsize[0] * 0.7, original_figsize[1] * 5/3))
    Ea_e = np.linspace(0.5, 0.8, 200)
    Ea_c = np.linspace(1, 2, 200)
    X, Y = np.meshgrid(Ea_e, Ea_c)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, sigma)
    Z = rv.pdf(pos)
    contour = axs.contourf(X, Y, Z, cmap='magma', levels=30)
    #axs.set_aspect(0.5)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    inset = fig.add_axes([0.13, 0.75, 0.03, 0.2])  
    colorbar = plt.colorbar(contour, cax=inset, orientation='vertical')
    colorbar.ax.yaxis.set_tick_params(colors='white')
    colorbar.set_ticks([0, np.max(Z)])
    colorbar.set_ticklabels(['0','7'],fontsize =14)
    axs.set_xlabel(r'$E_{ae}$ (eV)',fontsize = 14)
    axs.set_ylabel(r'$E_{ac}$ (eV)',fontsize = 14)
    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    axs.set_xticks([0.6,0.8])
    axs.set_xticklabels(['0.6','0.8'])
    axs.set_yticks([1.0,1.2,1.4,1.6,1.8,2.0])
    axs.set_yticklabels(['1.0','1.2','1.4','1.6','1.8','2.0'])
    return axs

def FigureCompHist(superim):
    T = [-20,0,20,40,60,80]
    T1 = [253,273,293,313,333,353]
    fig,axs = plt.subplots()
    Rel_T20 = np.empty((7,1))
    Rel_t_MCF = np.empty((7,1))
    Rel_allT_Rel = np.empty((7,1))
    Rel_allT_NSP = np.empty((7,1))
    Rel_data = np.empty((7,1))
    for i in range(len(T)):
        input_file = 'data/Rel_'+ str(T[i])
        Rel = np.genfromtxt(input_file, delimiter=',')
        Rel_data[i,0] = Rel[0,2]
        input_file = 'data/Rel_optimized_T=20_'+ str(T[i])
        Rel = np.genfromtxt(input_file, delimiter=',')
        Rel_T20[i,0] = Rel[0,2]
        input_file = 'data/Rel_allT_optimized_Rel_'+ str(T[i])
        Rel = np.genfromtxt(input_file, delimiter=',')
        Rel_allT_Rel[i,0] = Rel[0,2]
        #input_file = 'data/Rel_allT_optimized_NSP_'+ str(T[i])
        #Rel = np.genfromtxt(input_file, delimiter=',')
        #Rel_allT_NSP[i,0] = Rel[0,2]
        input_file = 'data/Rel_optimized_t_MCF_T=20_'+ str(T1[i])
        Rel = np.genfromtxt(input_file, delimiter=',')
        Rel_t_MCF[i] = Rel[0,2]
    input_file = 'data/Rel_allT'
    Rel = np.genfromtxt(input_file, delimiter=',')
    Rel_data[6,0] = Rel[0,2]
    input_file = 'data/Rel_allT_optimized_T=20'
    Rel = np.genfromtxt(input_file, delimiter=',')
    Rel_T20[6,0] = Rel[0,2]
    input_file = 'data/Rel_allT_optimized_Rel'
    Rel = np.genfromtxt(input_file, delimiter=',')
    Rel_allT_Rel[6,0] = Rel[0,2]
    #input_file = 'data/Rel_allT_optimized_NSP'
    #Rel = np.genfromtxt(input_file, delimiter=',')
    #Rel_allT_NSP[6,0] = Rel[0,2]
    input_file = 'data/Rel_allT_optimized_t_MCF_T=20'
    Rel = np.genfromtxt(input_file, delimiter=',')
    Rel_t_MCF[6,0] = Rel[0,2]
    dif_Rel_tMCF = (Rel_t_MCF - Rel_data)/Rel_data
    dif_Rel_T20 = (Rel_T20 - Rel_data)/Rel_data
    dif_Rel_allT = (Rel_allT_Rel - Rel_data)/Rel_data
    dif_Rel = np.concatenate((dif_Rel_allT,dif_Rel_T20,dif_Rel_tMCF),axis=1)
    bins = ['-20\u00B0C','0\u00B0C','20\u00B0C','40\u00B0C','60\u00B0C','80\u00B0C', 'all T']
    if superim == True:
        axs.bar(bins, dif_Rel_tMCF)
        axs.bar(bins, dif_Rel_T20)
        axs.bar(bins, dif_Rel_allT)
    else:
        x = np.arange(7)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        labels = ['$\epsilon_1$','$\epsilon_2$','$\epsilon_3$']
        for i in range(3):
            offset = width * multiplier
            axs.bar(x + offset, dif_Rel[:,i], width, label=labels[i])
            multiplier += 1
    axs.set_ylim(0,0.2)
    axs.set_ylabel('$\epsilon$ (Rel)')
    axs.set_xticks(x + width, bins)
    axs.legend(loc='upper left')
    axs.set_yticks([0.00,0.05,0.1,0.15,0.2])
    axs.set_yticklabels(['0.00','0.05','0.10','0.15','0.20'])
    return dif_Rel_T20,dif_Rel_tMCF,dif_Rel_allT

