import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from scipy.stats import norm

plt.style.use('ieee')
plt.rcParams['font.family'] = 'sans-serif'

main_directory = 'C:\\Users\\bysho\\OneDrive\\Escritorio\\Fernando\\Trabajo\\Reliability\\GARTNPUF\\'
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

    ax.scatter(data_fresh, data_fresh-data_aging, color='r', label='Aging')
    ax.scatter(data_aging, data_aging-data_thermal, color='g', label='Thermal')
    ax.scatter(data_thermal, data_thermal-data_electrical, color='y', label='Electrical')

    if i//2 == 1:
        ax.set_xlabel(r'Initial Current ($\mu$A)')
    if i%2 == 0:
        ax.set_ylabel(r'Current Shift ($\mu$A)')
    ax.text(0.025, 0.975, f'({chr(97 + i)}) ' + titles[i] , transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left', size = 7)

# Create a unique legend outside the axis
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.savefig('figures\\aged_devices_scatter.pdf')
#plt.show()
    
# file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\data\\Datos Gráfica.xlsx'
# sheet_name = 'Hoja1'
# df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
# data_gateV = df['GateV'].to_numpy()
# data_drainI = df[['IDVBG0V','IDVGB5V','IDVBG10V','IDVBG15V']].to_numpy()

# ## Figure for fresh devices: IV curves
# fig, axs = plt.subplots(figsize=(3.5,2.5))
# for i in range(4):
#     ax = axs
#     ax.plot(data_gateV, -data_drainI[:,i]*1e6, label=r'$V_{BG}$ = '+ str(int(5*i)) +' V')
# ax.set_xlabel(r'Top Gate Voltage, $V_{TG}$ (V)')
# ax.set_ylabel(r'Drain Current, $I_D$ ($\mu$ A)')
# ax.legend(loc='upper left')
# ax.grid(True)
# ax.set_xlim(10, -30)  # Flip the x-axis and set limits
# plt.tight_layout()
# plt.savefig('figures\\fresh_devices_IV_curves.pdf')
#plt.show()

print('Hello World')