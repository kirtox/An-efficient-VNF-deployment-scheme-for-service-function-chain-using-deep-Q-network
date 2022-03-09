# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 02:20:56 2020

@author: Wei-Cheng Wu
"""
import numpy as np
from config import VNFGroupConfig

group_config = VNFGroupConfig()

class Energy:
    def __init__(self):
        # setting
        self.number_of_nodes = 5
        self.number_of_vnfs = 5
        
        # after 3 rounds, node at IDLE state turn OFF
        # ex: IDLE time from -3 => -2 => -1, then turn OFF
        self.max_idle_time = -3
        
        # 60% of utilization 100%
        self.idle_energy = 0.6
        
        # sum of energy
        self.OIA_energy = 0

    
    def reset(self):
        self.OIA_energy = 0

    
    def update(self, used_nodes, state, idletime, vnf_collection, capaVNFs):
        # OIA
        
        # State change
        for i in range(self.number_of_nodes):
            # at IDLE state at present
            if state[i] == 'IDLE':
                # after 3 rounds, IDLE -> OFF
                if idletime[i] == 0:
                    state[i] = 'OFF'
                    capaVNFs[i] = [0, 0, 0, 0, 0]
                else:
                    idletime[i] += 1
                    if idletime[i] == 0:
                        state[i] = 'OFF'
                        capaVNFs[i] = [0, 0, 0, 0, 0]
            
            # at ACTIVE state
            # node is not used and is ACTIVE at present 
            if i not in used_nodes and state[i] == 'ACTIVE':
                state[i] = 'IDLE'
                idletime[i] = self.max_idle_time            

                
        # OFF -> ACTIVE or IDLE -> ACTIVE => These nodes is using. 
        for node, vnf in zip(used_nodes, vnf_collection):
            state[node] = 'ACTIVE'
            idletime[node] = 1
            
            # release VNFs
            #if capaVNFs[node][vnf] > 4:
            #    print("Capacity out of maximum capacity")
            
            if capaVNFs[node][vnf] == 4:
                #print(":::Warning: CapaVNF[{},{}] is full:::".format(str(node), str(vnf)))
                capaVNFs[node][vnf] = 0
                #print(":::Clean: CapaVNF[{},{}] is empty:::".format(str(node), str(vnf)))
            
            # VNF add
            capaVNFs[node][vnf] += 1
        
        #print("=============== Updated ===============")
 

    # evaluation of OIA energy
    def energy_OIA_mode(self, used_nodes, E_vnf, E_IDLEvnf, state, vnf_collection, capaVNFs):
        for node in range(self.number_of_nodes):
            if state[node] == 'OFF':
                continue
            elif state[node] == 'IDLE':
                # 4*(16+20+17+15+17) = 340 (Maximum energy consumption of each nodes) 
                #self.OIA_energy_withType += self.idle_energy*self.max_energy_of_each_node_withType
                self.OIA_energy += np.dot(capaVNFs[node], E_IDLEvnf)
                #for vnf in range(self.number_of_vnfs):
                #    self.OIA_energy += capaVNFs[node][vnf]*E_IDLEvnf[vnf]
            elif state[node] == 'ACTIVE':
                self.OIA_energy += np.dot(capaVNFs[node], E_vnf)
                #for vnf in range(self.number_of_vnfs):
                #    self.OIA_energy += capaVNFs[node][vnf]*E_vnf[vnf]  
        return self.OIA_energy
                    
         
if __name__ == '__main__':
    energy = Energy()
    # Energy consumption of each type of VNFs
    E_vnf = group_config.get_initialized_vnf()
    
    # IDLE Energy consumption of each type of VNFs
    E_IDLEvnf = group_config.get_initialized_IDLEvnf(energy.idle_energy)
    #E_IDLEvnf = group_config.get_initialized_IDLEvnf(E_vnf, energy.idle_energy)
    
    # state: ACTIVE, IDLE, OFF
    state = group_config.get_initialized_state()

    # There are five nodes. Each node have 5 type of VNFs and the number of each type is 4.
    capaVNFs = group_config.get_initialized_capaVNFs()
    
    # IDLE time of each node
    idletime = group_config.get_initialized_idletime()
        
    
    
    vnf_collection = [0,1,2,3,4]
    """
    used_nodes = [[4,2,3,1,2], [3,3,3,3,3], [2,3,1,1,3], [0,3,3,2,3], [1,4,3,1,3],
                  [4,3,3,1,2], [1,3,3,1,3], [4,1,3,1,3], [3,3,3,1,2], [3,1,3,1,3],
                  [1,3,3,1,3], [1,3,3,1,1], [1,3,0,1,3], [0,4,2,0,1], [4,4,3,3,2],
                  [4,4,3,3,1], [0,2,4,1,2], [4,3,2,3,1], [1,1,4,1,1], [1,1,1,1,1]]
    """
    used_nodes = [[4,2,3,1,2], [3,3,3,3,3], [3,3,3,3,3], [3,3,3,3,3], [3,3,3,3,3]]
    #used_nodes = [[4,4,3,3,1], [0,2,4,1,2], [4,3,2,3,1]]
    #print("Input: ", used_nodes)
    for i in range(5):
        print("===============Round", i, "start===============")
        print("VNF No.:      0  1  2  3  4")
        print("Used nodes: ", used_nodes[i])
        energy.update(used_nodes[i], state, idletime, vnf_collection, capaVNFs)
        print("State:                   ", state)
        print("Energy of each VNF:      ", E_vnf)
        print("IDLE Energy of each VNF: ", E_IDLEvnf)
        print("\nAllocate_of_eachVNF:")
        print("VNF:")
        print("  0 1 2 3 4")
        print(capaVNFs)
        print("IDLE time countdown: ", idletime)
        #if i != 0 and i % 10 == 0:
        #    energy3.release_capacity(capacity)
        #    print("=====Release success=====")
        print("\nEnergy consumption:")
        print("    OIA: {:>12g}".format(round(energy.energy_OIA_mode(used_nodes, E_vnf, E_IDLEvnf, state, vnf_collection, capaVNFs), 0)))
   
        print("===============Round ", i, "end===============")
        print("\n")
        
    #print("State: ", energy3.state)
    #print("Allocate: ", energy3.capacity)
    #print("IDLE time countdown: ", energy3.idletime)
    ##print("OIA: ", energy3.energy_OIA_mode(E, state, idletime, capacity))

