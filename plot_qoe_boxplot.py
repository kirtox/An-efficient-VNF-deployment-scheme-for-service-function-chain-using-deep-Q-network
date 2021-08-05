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

fig, ax = plt.subplots(figsize=(10, 7), dpi=600)

# Random
y1 = smooth(list(df[0]))
RAND_Q1, RAND_Q2, RAND_Q3 = eval_Q(sorted(list(df[0])))

print("Random ==>")
print("Q1: ", round(RAND_Q1, 2), ", Q2: ", round(RAND_Q2, 2), ", Q3: ", round(RAND_Q3, 2), "\n")

# Brute Force qoe
y2 = smooth(list(df[5]))
BFQ_Q1, BFQ_Q2, BFQ_Q3 = eval_Q(sorted(list(df[5])))

print("Brute Force qoe ==>")
print("Q1: ", round(BFQ_Q1, 2), ", Q2: ", round(BFQ_Q2, 2), ", Q3: ", round(BFQ_Q3, 2), "\n")

# Brute Force energy
y3 = smooth(list(df[10]))
BFE_Q1, BFE_Q2, BFE_Q3 = eval_Q(sorted(list(df[10])))

print("Brute Force energy ==>")
print("Q1: ", round(BFE_Q1, 2), ", Q2: ", round(BFE_Q2, 2), ", Q3: ", round(BFE_Q3, 2), "\n")

# DQN-Q2-SFC
y4 = smooth(list(df_QQ[0]))
Q2SFC_Q1, Q2SFC_Q2, Q2SFC_Q3 = eval_Q(sorted(list(df_QQ[0])))

print("DQN-Q2-SFC ==>")
print("Q1: ", round(Q2SFC_Q1, 2), ", Q2: ", round(Q2SFC_Q2, 2), ", Q3: ", round(Q2SFC_Q3, 2), "\n")

# DQN-QQE
y5 = smooth(list(df[15]))
QQE_Q1, QQE_Q2, QQE_Q3 = eval_Q(sorted(list(df[15])))

print("DQN-QQE ==>")
print("Q1: ", round(QQE_Q1, 2), ", Q2: ", round(QQE_Q2, 2), ", Q3: ", round(QQE_Q3, 2), "\n")

#print("DQN compare: ")
#print(sum(list(df_QQ[0]))/len(list(df_QQ[0])), sum(list(df[5]))/len(list(df[5])))

bplot = plt.boxplot([y1, y2, y3, y4, y5], labels = ['Random', 'Brute Force-qoe', 'Brute Force-energy', 'DQN-Q2-SFC', 'DQN-QQE'], patch_artist = True)

MinMax = [item.get_ydata()[1] for item in bplot['whiskers']]
print("Before: ", MinMax)
MinMax = [round(item, 2) for item in MinMax]
print("After: ", MinMax)



colors = ['#DB2763', '#B0DB43', '#12EAEA', '#BCE7FD', '#C492B1']

# colors
for patch, color in zip(bplot['boxes'], colors): 
    patch.set_facecolor(color)

ax.grid(axis='y', linestyle='--')
ax.set_axisbelow(True)
ax.set_title('Comparison of QoE')
#plt.legend(['Random', 'Brute Force-qoe', 'Brute Force-energy', 'DQN-Q2-SFC', 'DQN-QQE'], loc=2, prop={'size': 20}, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
#plt.xlabel('Iteration')
ax.set_ylabel('QoE')
#plt.savefig('Comparison_qoe.png')
plt.show()
