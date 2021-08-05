# An-Efficient-VNF-Deployment-Mechanism-for-SFC-in-5G-using-Deep-Q-Network

***

# File Descriptions

## DQN-QQE 
- main.py
	- It is the main function that executes DQN-QQE. 
	- Moreover, random and brute force are also executed in main.py.

- env.py
	- This file sets the environment of DQN-QQE.

- dqn.py
	- There are the neural network structure and lots of DQN functions.

- config.py
	- It sets lots of parameters.
	- DQN-QQE and DQN-Q2-SFC use the same file.
 

## DQN-Q2-SFC
- main_QQ.py
	- It is the main function that executes DQN-QQE. 
	- Moreover, random and brute force are also executed in main.py.

- env_QQ.py
	- This file sets the environment of DQN-QQE.

- dqn_QQ.py
	- There are the neural network structure and lots of DQN functions.

- config.py
	- It sets lots of parameters.
	- DQN-QQE and DQN-Q2-SFC use the same file.


## Random
- random_sfc.py
	- Random scheme.

## Brute Force
- brute_sfc_energy.py
	- Brute force approach on energy aspect.

- brute_sfc_qoe.py
	- Brute force approach on qoe aspect.

***

# Execute
## Run run.bat
- It is a batch file.
- Line 1:
	- call C:\Users\SNMLAB\Anaconda3\Scripts\activate.bat C:\Users\SNMLAB\Anaconda3
	- It need to know where the activate.bat of Anaconda3 is.
- Line 2:
	- python main.py
	- Run DQN-QQE, random, and brute force.
- Line 3:
	- python main_QQ.py
	- Run DQN-Q2-SFC
- Line 4:
	- pause
	- Stop batch file.

***

# Plots

## Quality of Experience (QoE)
![image](https://github.com/kirtox/An-Efficient-VNF-Deployment-Mechanism-for-SFC-in-5G-using-Deep-Q-Network/blob/master/Comparison_boxplot_qoe_1500iters.png?raw=true)

- plot_qoe.py
	- Line chart

- plot_qoe_boxplot.py
	- Boxplot chart

## Error rate
![image](https://github.com/kirtox/An-Efficient-VNF-Deployment-Mechanism-for-SFC-in-5G-using-Deep-Q-Network/blob/master/Comparison_boxplot_error_1500iters.png?raw=true)

- plot_error.py
	- Line chart

- plot_error_boxplot.py
	- Boxplot chart

## Maximum energy consumption
![image](https://github.com/kirtox/An-Efficient-VNF-Deployment-Mechanism-for-SFC-in-5G-using-Deep-Q-Network/blob/master/Comparison_boxplot_max_energy_1500iters.png?raw=true)

- plot_max_energy.py
	- Line chart

- plot_max_energy_boxplot.py
	- Boxplot chart

## Average energy consumption
![image](https://github.com/kirtox/An-Efficient-VNF-Deployment-Mechanism-for-SFC-in-5G-using-Deep-Q-Network/blob/master/Comparison_boxplot_mean_energy_1500iters.png?raw=true)

- plot_mean_energy.py
	- Line chart

- plot_mean_energy_boxplot.py
	- Boxplot chart

## Average processing time
![image](https://github.com/kirtox/An-Efficient-VNF-Deployment-Mechanism-for-SFC-in-5G-using-Deep-Q-Network/blob/master/Comparison_bar_mean_time_1500iters.png?raw=true)

- plot_time.py
	- Line chart

- plot_time_barplot.py
	- Bar chart

# Using Ant Colony Optimization algorithm (ACO) to iris dataset
- To get the clusters
	- ACO.ipynb


***
