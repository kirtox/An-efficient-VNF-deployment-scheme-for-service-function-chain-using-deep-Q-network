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

def eval_Q(df):
    iterations = 1500
    loc_Q1 = (1*(iterations+1)/4) - 1
    loc_Q2 = (iterations/2)- 1
    loc_Q3 = (3*(iterations+1)/4) - 1
    #print("Q1: ", loc_Q1, ", Q2: ", loc_Q2, ", Q3: ", loc_Q3)
    #print(int(loc_Q1))
    Q1 = 0
    Q2 = 0
    Q3 = 0

    # Q1
    Q1 = df[int(loc_Q1)]*(loc_Q1-int(loc_Q1)) + df[int(loc_Q1)+1]*(1-(loc_Q1-int(loc_Q1)))
   
    # Q2
    if iterations%2 == 0:
        Q2 = (df[int(loc_Q2)] + df[int(loc_Q2)+1]) / 2
    else:
        Q2 = df[int(loc_Q2)+1]
    
    # Q3
    Q3 = df[int(loc_Q3)]*(loc_Q3-int(loc_Q3)) + df[int(loc_Q3)+1]*(1-(loc_Q3-int(loc_Q3)))
    
    return Q1, Q2, Q3

def smooth(sequence):
    y = sequence[:N-1]
    for i in range(len(sequence)-N+1):
        y.append(np.mean(sequence[i: i+N]))
    return y

df = pd.read_csv('output.txt', header=None)
df_QQ = pd.read_csv('output_QQ.txt', header=None)

x = range(df.shape[0])

plt.figure(figsize=(10, 7), dpi=600)

# Random
y1 = smooth(list(df[0]))
plt.plot(x, y1, c='#DB2763', linewidth=2)

# Brute Force qoe
y2 = smooth(list(df[5]))
plt.plot(x, y2, c='#B0DB43', linewidth=2)

# Brute Force energy
y3 = smooth(list(df[10]))
plt.plot(x, y3, c='#12EAEA', linewidth=2)

# DQN-Q2-SFC
y4 = smooth(list(df_QQ[0]))
plt.plot(x, y4, c='#BCE7FD', linewidth=2)

# DQN-QQE
y5 = smooth(list(df[15]))
plt.plot(x, y5, c='#C492B1', linewidth=2)


plt.grid()
plt.title('Comparison of QoE')
plt.legend(['Random', 'Brute Force-qoe', 'Brute Force-energy', 'DQN-Q2-SFC', 'DQN-QQE'], loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.xlabel('Iteration')
plt.ylabel('QoE')
#plt.savefig('Comparison_qoe.png')
plt.show()
