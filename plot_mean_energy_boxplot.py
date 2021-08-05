# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:58:20 2020

@author: Wei-Cheng Wu
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

smooth_rate = 0.1
N = int(1/smooth_rate)


def smooth(sequence):
    y = sequence[:N-1]
    for i in range(len(sequence)-N+1):
        y.append(np.mean(sequence[i: i+N]))
    return y

df = pd.read_csv('output.txt', header=None)
df_QQ = pd.read_csv('output_QQ.txt', header=None)

x = range(df.shape[0])

fig, ax = plt.subplots(figsize=(10, 7), dpi=600)

# Random
y1 = smooth(list(df[2]))
tmp_y1 = np.mean(list(df[2]))

# Brute Force qoe
y2 = smooth(list(df[7]))
tmp_y2 = np.mean(list(df[7]))

# Brute Force energy
y3 = smooth(list(df[12]))
tmp_y3 = np.mean(list(df[12]))

# DQN-Q2-SFC
y4 = smooth(list(df_QQ[2]))
tmp_y4 = np.mean(list(df_QQ[2]))

# DQN-QQE
y5 = smooth(list(df[17]))
tmp_y5 = np.mean(list(df[17]))

bplot = ax.boxplot([y1, y2, y3, y4, y5], labels = ['Random', 'Brute Force-qoe', 'Brute Force-energy', 'DQN-Q2-SFC', 'DQN-QQE'], patch_artist = True)

colors = ['#DB2763', '#B0DB43', '#12EAEA', '#BCE7FD', '#C492B1']
# colors
for patch, color in zip(bplot['boxes'], colors): 
    patch.set_facecolor(color)

print(tmp_y4, '&', tmp_y5)

ax.grid(axis='y', linestyle='--')
ax.set_axisbelow(True)
ax.set_title('Comparison of Average Energy')
#plt.legend(['Random', 'Brute Force-qoe', 'Brute Force-energy', 'DQN-Q2-SFC', 'DQN-QQE'], loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
#plt.xlabel('Iteration')
ax.set_ylabel('Energy Consumption (J)')
#plt.savefig('Comparison_mean_energy_300iters.png')
plt.show()