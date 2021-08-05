import tensorflow as tf
tf.compat.v1.reset_default_graph()

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from env_QQ import VNFGroup_QQ
from dqn_QQ import DQN_QQ

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env_QQ = VNFGroup_QQ()
    agent_QQ = DQN_QQ()
    agent_QQ.load()
    
    times = 1000
    
    for iteration in range(times):
        #print("========================== QQ Iteration {} ==========================".format(iteration))
    
        env_QQ.reset()

        # DQN eval
        #sfc_requests_energy = env_QQ.sfc_requests_energy
        
        #print("\n\nsfc_requests_energy: \n", sfc_requests_energy)
        
        start = time.time()
        sfc_requests = env_QQ.sfc_requests[:]
        print("\n\nsfc_requests: \n", sfc_requests)
        
        #print("SFC: ", sfc_requests)
        
        dqn_error = 0.0
        [B_, D_] = sfc_requests.pop(0)
        #E_ = sfc_requests_energy.pop(0)
    
        print("B_, D_: ", [B_, D_])
        #print("E_: ", E_)
       
        total_OIA = []
            
        while True:
            
            observation = env_QQ.start(B_, D_)
            #state_S = env_QQ.S.copy()
            for n in range(5):
                #print("n_id: ", n)
                action = agent_QQ.choose_action(observation, larger_greedy=1.0)
                observation_, reward, done, info = env_QQ.step(action)
                
                #print("INFO: ", info)
                
                if info['id']:  dqn_error += 1
                if done:  break
                observation = observation_
            #energy_all_ACTIVE = env_QQ.energy_all_ACTIVE
            #energy_OIA = env_QQ.energy_OIA
            total_OIA.append(round(env_QQ.energy_OIA, 2))
            
            try:
                [B_, D_] = sfc_requests.pop(0)
                #E_ = sfc_requests_energy.pop(0)
                
                print("B_, D_: ", [B_, D_])
            except IndexError:
                break

        end = time.time()
        processing_time = round((end-start), 2)
        
        # Processing time
        print("Period: {:g}s".format(processing_time))
        
        # Energy assign
        mean_energy_OIA = sum(total_OIA)/env_QQ.num_requests
        max_energy_OIA = max(total_OIA)
        
        # QoE & Error rate
        total_qoe = env_QQ.get_mean_qoe()
        error_rate = dqn_error/env_QQ.num_requests
        
        # output_QQ format
        # ===================================================================================================
        # qoe        : dqn=>0
        # error      : dqn=>1
        # mean energy: dqn=>2
        # max energy : dqn=>3
        # time       : dqn=>4
        # ===================================================================================================
        
        with open('output_QQ.txt', 'a') as f:
            f.write('{0}, {1}, {2}, {3}, {4}\n'.format(
                total_qoe, error_rate,
                mean_energy_OIA, max_energy_OIA, processing_time)
            )
          
        #print(env_QQ.B)

        # DQN train
        print("=======================DQN train=======================")
        sfc_requests = env_QQ.sfc_requests[:]
        env_QQ.reset(use_sfc_requests=sfc_requests)
        [B_, D_] = sfc_requests.pop(0)
        #print("Train B_, D_: ", [B_, D_])
        while True:
            observation = env_QQ.start(B_, D_)
            for n in range(5):
                #print("Train_n: ", n)
                action = agent_QQ.choose_action(observation)
                observation_, reward, done, info = env_QQ.step(action)
                agent_QQ.store_transition(observation, action, reward, observation_)
                #print("INFO: ", info)
                if info['id']:  dqn_error += 1
                if done:  break
                observation = observation_
            agent_QQ.learn()
            try:
                [B_, D_] = sfc_requests.pop(0)
                #print("Train B_, D_: ", [B_, D_])
            except IndexError:
                break

        agent_QQ.save()
    
