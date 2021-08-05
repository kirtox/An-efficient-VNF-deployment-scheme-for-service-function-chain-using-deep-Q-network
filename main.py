import tensorflow as tf
tf.compat.v1.reset_default_graph()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from env import VNFGroup
from dqn import DQN
from random_sfc import RandomSFC
from brute_sfc_qoe import BruteSFC_QoE
from brute_sfc_energy import BruteSFC_Energy

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env = VNFGroup()
    agent = DQN()
    agent.load()
    random_sfc = RandomSFC()
    brute_sfc_qoe = BruteSFC_QoE()
    brute_sfc_energy = BruteSFC_Energy()

    # 12/19 如果1000 iters還是不好 試試看把agent.load()放在33行
    times = 1000
    
    #while True:
    for iteration in range(times):
        #agent.load()
        print("========================== QQE Iteration {} ==========================".format(iteration))

        env.reset()
        random_sfc.reset()
        brute_sfc_qoe.reset()
        brute_sfc_energy.reset()
        
        # Random
        # ===================================================================================================
        print("=============== Random Start ===============")
        random_sfc.set_sfc_requests(env.sfc_requests[:])
        random_sfc.select()
        print("=============== Random End ===============")
        # ===================================================================================================
        
        # BruteForce for qoe
        # ===================================================================================================
        print("=============== BruteForce_QoE Start ===============")
        brute_sfc_qoe.set_sfc_requests(env.sfc_requests[:])
        brute_sfc_qoe.select()
        print("=============== BruteForce_QoE End ===============")
        # ===================================================================================================
        
        # BruteForce for energy
        # ===================================================================================================
        print("=============== BruteForce_Energy Start ===============")
        brute_sfc_energy.set_sfc_requests(env.sfc_requests[:])
        brute_sfc_energy.select()
        print("=============== BruteForce_Energy End ===============")
        # ===================================================================================================
        

        # DQN eval
        #sfc_requests_energy = env.sfc_requests_energy
        
        #print("\n\nsfc_requests_energy: \n", sfc_requests_energy)
        start = time.time()
        sfc_requests = env.sfc_requests[:]
        print("\n\nsfc_requests: \n", sfc_requests)
        tmp_src_sfc = sfc_requests.copy()
        
        #print("SFC: ", sfc_requests)
        
        dqn_error = 0.0
        [B_, D_] = sfc_requests.pop(0)
        #E_ = sfc_requests_energy.pop(0)
    
        print("B_, D_: ", [B_, D_])
        #print("E_: ", E_)
        
        total_OIA = []
        
            
        while True:
            txt = []
            
            observation = env.start(B_, D_)
            #state_S = env.S.copy()
            for n in range(5):
                #print("n_id: ", n)
                action = agent.choose_action(observation, larger_greedy=1.0)
                observation_, reward, done, info = env.step(action)
                
                #print("INFO: ", info)
                
                if info['id']:  dqn_error += 1
                if done:  break
                observation = observation_

            total_OIA.append(round(env.energy_OIA, 0))
            
            try:
                [B_, D_] = sfc_requests.pop(0)
                #E_ = sfc_requests_energy.pop(0)
    
                print("B_, D_: ", [B_, D_])
                #print("E_: ", E_)
                #print("TRY")
                #print("B_, D_: ", [B_, D_])
            except IndexError:
                break
        end = time.time()
        
        print("Period: {:g}s".format(round((end-start), 2)))
        
        #sum_all_ACTIVE = total_all_ACTIVE[-1]
        #sum_OIA = total_OIA[-1]
        mean_OIA = sum(total_OIA)/env.num_requests
        max_OIA = max(total_OIA)
        
        # output format
        # ===================================================================================================
        # qoe        : random=>0 , brute_qoe=>5 , brute_energy=>10 ,QQE=>15
        # error      : random=>1 , brute_qoe=>6 , brute_energy=>11 ,QQE=>16
        # mean energy: random=>2 , brute_qoe=>7 , brute_energy=>12 ,QQE=>17
        # max energy : random=>3 , brute_qoe=>8 , brute_energy=>13 ,QQE=>18
        # time       : random=>4 , brute_qoe=>9 , brute_energy=>14 ,QQE=>19
        # ===================================================================================================
        
        with open('output.txt', 'a') as f:
            f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}\n'.format(
                random_sfc.get_mean_qoe(), random_sfc.get_error_rate(),
                random_sfc.get_mean_energy_OIA(), random_sfc.get_max_energy_OIA(), 
                random_sfc.time,
                brute_sfc_qoe.get_mean_qoe(), brute_sfc_qoe.get_error_rate(),
                brute_sfc_qoe.get_mean_energy_OIA(), brute_sfc_qoe.get_max_energy_OIA(), 
                brute_sfc_qoe.time,
                brute_sfc_energy.get_mean_qoe(), brute_sfc_energy.get_error_rate(),
                brute_sfc_energy.get_mean_energy_OIA(), brute_sfc_energy.get_max_energy_OIA(), 
                brute_sfc_energy.time,
                env.get_mean_qoe(), dqn_error/env.num_requests,
                mean_OIA, max_OIA,
                round((end-start), 2))
            )
        
        #print(env.B)

        # DQN train
        print("=======================DQN train=======================")
        sfc_requests = env.sfc_requests[:]
        env.reset(use_sfc_requests=sfc_requests)
        [B_, D_] = sfc_requests.pop(0)
        #print("Train B_, D_: ", [B_, D_])
        while True:
            observation = env.start(B_, D_)
            for n in range(5):
                #print("Train_n: ", n)
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                agent.store_transition(observation, action, reward, observation_)
                #print("INFO: ", info)
                if info['id']:  dqn_error += 1
                if done:  break
                observation = observation_
            agent.learn()
            try:
                [B_, D_] = sfc_requests.pop(0)
                #print("Train B_, D_: ", [B_, D_])
            except IndexError:
                break

        agent.save()

