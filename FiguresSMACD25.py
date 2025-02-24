import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from scipy.stats import norm

plt.style.use('ieee')
plt.rcParams['font.family'] = 'sans-serif'

file_path_1 = 'data\\Taula 1 - IDVD LIN BACKWARD +5V.xlsx'
file_path_2 = 'data\\Taula 2 - IDVD SAT BACKWARD +5V.xlsx'
file_path_3 = 'data\\Taula 3 - IDVG SAT IV1 FORWARD -20V.xlsx'
file_path_4 = 'data\\Taula 4 - IDVG LIN IV1 FORWARD -18V.xlsx'
file_path_array = [file_path_1,file_path_2,file_path_3,file_path_4]
sheet_name = 'Hoja1'
effects = ['aging','thermal','electrical','all']

data_fresh_devices = np.zeros((75,4))
for data in [1,2,3,4]:
    file_path = 'C:\\Users\\Usuario\\Desktop\\GARTNPUF\\' + file_path_array[data-1]
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    data_fresh_devices[:,data-1] = df['T0'].to_numpy()
    data_currents_aging = df[['T0', 'T1']].to_numpy()
    data_currents_thermal = df[['T1', 'T2']].to_numpy()
    data_currents_electrical = df[['T2', 'T3']].to_numpy()
    data_currents_all = df[['T0','T1','T2','T3']].to_numpy()

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
plt.show()
print('Hello World')