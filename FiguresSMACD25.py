import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from scipy.stats import norm

plt.style.use('ieee')
plt.rcParams['font.family'] = 'sans-serif'

#main_directory = 'C:\\Users\\bysho\\OneDrive\\Escritorio\\Fernando\\Trabajo\\Reliability\\GARTNPUF\\'
main_directory = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\'
file_path_1 = 'data\\Taula 1 - IDVD LIN BACKWARD +5V.xlsx'
file_path_2 = 'data\\Taula 2 - IDVD SAT BACKWARD +5V.xlsx'
file_path_3 = 'data\\Taula 3 - IDVG SAT IV1 FORWARD -20V.xlsx'
file_path_4 = 'data\\Taula 4 - IDVG LIN IV1 FORWARD -18V.xlsx'
file_path_array = [file_path_1,file_path_2,file_path_3,file_path_4]
sheet_name = 'Hoja1'
effects = ['aging','thermal','electrical','all']

data_fresh_devices = np.zeros((75,4))
data_currents_aging = np.zeros((75,4))
data_currents_thermal = np.zeros((75,4))
data_currents_electrical = np.zeros((75,4))
for data in [1,2,3,4]:
    file_path = main_directory + file_path_array[data-1]
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    data_fresh_devices[:,data-1] = df['T0'].to_numpy()
    data_currents_aging[:,data-1] = df['T1'].to_numpy()
    data_currents_thermal[:,data-1] = df['T2'].to_numpy()
    data_currents_electrical[:,data-1] = df['T3'].to_numpy()
    #data_currents_all = df[['T0','T1','T2','T3']].to_numpy()

## Figure for fresh devices: Histograms
fig, axs = plt.subplots(2, 2, figsize=(3.5,3.3))
titles = ['LB','SB','SF','LF']
for i in range(4):
    ax = axs[i//2,i%2]
    data = data_fresh_devices[:,i]
    mean, std = norm.fit(data)
    ax.hist(data, bins=10, density=True, alpha=0.5, color='b', edgecolor='black')
    
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    ax.plot(x, p, 'k', linewidth=2)
    
    #ax.legend(loc='upper right')
    #ax.set_title(titles[i])
    #ax.text(0.95, 0.95, f'$\mu$: {mean:.2f} $\mu$A \n$\sigma$: {std:.2f} $\mu$A', transform=ax.transAxes, 
    #        verticalalignment='top', horizontalalignment='right',size=8)
    if i//2 == 0:
        ax.set_xlabel('')
    else:
        ax.set_xlabel(r'Current ($\mu$A)')
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.text(0.025, 0.975, f'({chr(97 + i)}) ' + titles[i] + f'\n $\mu$: {mean:.1f} $\mu$A \n$\sigma$: {std:.1f} $\mu$A', transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left', size = 7)
plt.tight_layout()
plt.savefig('figures\\fresh_devices_histograms.pdf')
#plt.show()

## Figure for aged devices: Histograms
fig, axs = plt.subplots(2, 2, figsize=(3.5,3.3))
titles = ['LB','SB','SF','LF']
for i in range(4):
    ax = axs[i//2,i%2]
    data_fresh = data_fresh_devices[:,i]
    data_aging = data_currents_aging[:,i]
    data_thermal = data_currents_thermal[:,i]
    data_electrical = data_currents_electrical[:,i]

    ax.hist(data_fresh, bins=10, density=True, alpha=0.5, color='b', edgecolor='black', label='Fresh')
    ax.hist(data_aging, bins=10, density=True, alpha=0.5, color='r', edgecolor='black', label='Aging')
    ax.hist(data_thermal, bins=10, density=True, alpha=0.5, color='g', edgecolor='black', label='Thermal')
    ax.hist(data_electrical, bins=10, density=True, alpha=0.5, color='y', edgecolor='black', label='Electrical')

    
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    
    ax.legend(loc='upper right')
    #ax.set_title(titles[i])
    #ax.text(0.95, 0.95, f'$\mu$: {mean:.2f} $\mu$A \n$\sigma$: {std:.2f} $\mu$A', transform=ax.transAxes, 
    #        verticalalignment='top', horizontalalignment='right',size=8)
    if i//2 == 0:
        ax.set_xlabel('')
    else:
        ax.set_xlabel(r'Current ($\mu$A)')
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.text(0.025, 0.975, f'({chr(97 + i)}) ' + titles[i] , transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left', size = 7)
plt.tight_layout()
plt.savefig('figures\\aged_devices_histograms.pdf')
#plt.show()

# Figure for aged devices: Scatter plots
fig, axs = plt.subplots(2, 2, figsize=(3.5,3.3))
titles = ['LB','SB','SF','LF']
for i in range(4):
    ax = axs[i//2,i%2]
    data_fresh = data_fresh_devices[:,i]
    data_aging = data_currents_aging[:,i]
    data_thermal = data_currents_thermal[:,i]
    data_electrical = data_currents_electrical[:,i]

    ax.scatter(data_fresh, data_fresh-data_aging, color='r', edgecolor='black', label='Off time', s=10, alpha=0.6, linewidths=0.3)
    ax.scatter(data_aging, data_aging-data_thermal, color='g', edgecolor='black', label='Thermal', s=10, alpha=0.6, linewidths=0.3, marker='^')
    ax.scatter(data_thermal, data_thermal-data_electrical, color='y', edgecolor='black', label='Bias stress', s=10, alpha=0.6, linewidths=0.3, marker='s')
    
    ax.grid(True, alpha=0.3)  # Make the grid transparent
    if i//2 == 1:
        ax.set_xlabel(r'Initial Current, $I_{in}$ ($\mu$A)')
    if i%2 == 0:
        ax.set_ylabel(r'Current Shift, $\Delta I$ ($\mu$A)')
    ax.text(0.025, 0.975, f'({chr(97 + i)}) ' + titles[i] , transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left', size = 7)

# Create a unique legend outside the axis
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.075))

plt.tight_layout()
plt.savefig('figures\\aged_devices_scatter.pdf', bbox_inches='tight')
#plt.show()
    
file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\data\\IdVdlin4.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
data_drainI_lin = df['DrainI'].to_numpy()
data_drainV_lin = df[['DrainV']].to_numpy()

file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\data\\IdVdsat4.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
data_drainI_sat = df['DrainI'].to_numpy()
data_drainV_sat = df[['DrainV']].to_numpy()

file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\data\\IdVglin4.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
data_drainI_lin_1 = df['DrainI'].to_numpy()
data_gateV_lin = df[['GateV']].to_numpy()

file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\data\\IdVgsat4.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
data_drainI_sat_1 = df['DrainI'].to_numpy()
data_gateV_sat = df[['GateV']].to_numpy()

## Figure for fresh devices: IV curves
fig, axs = plt.subplots(1,2,figsize=(3.5,2.5))

axs[0].plot(data_drainV_lin, data_drainI_lin*1e6, 'b-', label=r'$V_{TG}$ = -20 V') 
axs[0].plot(data_drainV_sat, data_drainI_sat*1e6, 'r--', label=r'$V_{TG}$ = -6 V')
axs[0].scatter(data_drainV_lin[0], data_drainI_lin[0]*1e6, color='b', s=20, alpha=1, linewidths=0.3)
axs[0].scatter(data_drainV_sat[0], data_drainI_sat[0]*1e6, color='r', s=20, alpha=1, linewidths=0.3)
axs[0].text(data_drainV_lin[0] - 1, data_drainI_lin[0]*1e6 + 2, 'LB', fontsize=8, verticalalignment='top', horizontalalignment='right')
axs[0].text(data_drainV_sat[0], data_drainI_sat[0]*1e6, 'SB', fontsize=8, verticalalignment='bottom', horizontalalignment='right')

axs[1].plot(data_gateV_lin, data_drainI_lin_1*1e6, 'g-.', label=r'$V_{D}$ = -1 V') 
axs[1].plot(data_gateV_sat, data_drainI_sat_1*1e6, 'm:', label=r'$V_{D}$ = -15 V')
axs[1].scatter(data_gateV_lin[231], data_drainI_lin_1[231]*1e6, color='g', s=20, alpha=1, linewidths=0.3)
axs[1].scatter(data_gateV_sat[-1], data_drainI_sat_1[-1]*1e6, color='m', s=20, alpha=1, linewidths=0.3)
axs[1].text(data_gateV_lin[231], data_drainI_lin_1[231]*1e6 + 1, 'LF', fontsize=8, verticalalignment='bottom', horizontalalignment='left')
axs[1].text(data_gateV_sat[-1] + 1, data_drainI_sat_1[-1]*1e6 + 2, 'SF', fontsize=8, verticalalignment='top', horizontalalignment='left')

axs[0].set_xlabel(r'Drain Voltage, $V_{D}$ (V)')
axs[1].set_xlabel(r'Top-Gate Voltage, $V_{TG}$ (V)')
axs[0].set_ylabel(r'Drain Current, $I_D$ ($\mu$ A)')
axs[1].set_yticks([0, -10, -20, -30])
axs[0].legend(loc='upper left', fontsize='small', bbox_to_anchor=(0, 0.9))
axs[1].legend(loc='lower right', fontsize='small', bbox_to_anchor=(1, 0.1))
axs[0].grid(True)
axs[1].grid(True)
#axs[0].set_xlim(10, -30)  # Flip the x-axis and set limits

plt.tight_layout()
plt.savefig('figures\\devices_IV_curves.pdf')
plt.show()

print('Hello World')