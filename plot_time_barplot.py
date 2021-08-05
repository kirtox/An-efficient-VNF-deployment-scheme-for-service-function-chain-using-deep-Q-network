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
#y1 = smooth(list(df[4]))
y1 = np.mean(list(df[4]))
bplot1 = ax.bar(['Random'], [y1], color='#DB2763', hatch='o' ,width=0.5)

# Brute Force qoe
y2 = np.mean(list(df[9]))
bplot2 = ax.bar(['Brute Force-qoe'], [y2], color='#B0DB43', hatch='+' ,width=0.5)

# Brute Force energy
y3 = np.mean(list(df[14]))
bplot3 = ax.bar(['Brute Force-energy'], [y3], color='#12EAEA', hatch='.' ,width=0.5)

# DQN-Q2-SFC
y4 = np.mean(list(df_QQ[4]))
bplot4 = ax.bar(['DQN-Q2-SFC'], [y4], color='#BCE7FD', hatch='\\' ,width=0.5)

# DQN-QQE
y5 = np.mean(list(df[19]))
bplot5 = ax.bar(['DQN-QQE'], [y5], color='#C492B1', hatch='/' ,width=0.5)

#colors = ['darkgreen', 'darkmagenta', 'darkcyan', 'darkred', 'darkblue']
#hatchs = ('/', '\\', 'o', '-', '+')
#bplot = plt.bar(['Random', 'Brute Force-qoe', 'Brute Force-energy', 'DQN-Q2-SFC', 'DQN-QQE'], [y1, y2, y3, y4, y5], color=colors, hatch=hatchs ,width=0.5)
#print(y4, '&', y5)
#print(y2, '&', y3)
print("Random: ", round(y1, 2))
print("Brute Force-qoe: ", round(y2, 2), ", Brute Force-energy: ", round(y3, 2))
print("DQN-Q2-SFC: ", round(y4, 2), ", DQN-QQE: ", round(y5, 2))

ax.grid(axis='y', linestyle='--')
ax.set_axisbelow(True)
ax.set_title('Comparison of Average Processing Time')
#plt.legend(['Random', 'Brute Force-qoe', 'Brute Force-energy', 'DQN-Q2-SFC', 'DQN-QQE'], loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
#plt.xlabel('Iteration')
ax.set_ylabel('second (s)')
#plt.savefig('Comparison_error.png')
plt.show()
