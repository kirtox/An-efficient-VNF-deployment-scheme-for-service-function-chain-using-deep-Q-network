# coding=utf-8
import numpy as np
from config import VNFGroupConfig
from energy import Energy

group_config = VNFGroupConfig()
energy = Energy()

P = 10.0


class VNFGroup:
    def __init__(self):
        self.B = None            # B: 可用頻寬,10-20G
        self.D = None            # D: link延遲,10-20ms
        self.S = None            # B, B_, D, D_, Dc, action_index
        self.E = None            # E: energy consumption of each node
        self.E_vnf = None        # E_vnf: energy consumption of each type of VNFs
        self.E_IDLEvnf = None    # E_IDLEvnf: IDLE energy consumption of each type of VNFs
        self.state = None
        self.capacity = None
        self.idletime = None
        
        self.num_requests = 100  # 一次episode訓練接受的sfc請求數
        self.B_min = 16          # 所需頻寬範圍最小值,單位MB
        self.B_max = 256         # 所需頻寬範圍最大值,單位MB
        self.D_min = 50          # 最大延遲約束最小值,單位ms,>40ms
        self.D_max = 90          # 最大延遲約束最大值,單位ms
        self.E_min = 300
        self.E_max = 400

        
        self.sfc_requests = None
        self.sfc_requests_energy = None
        self.running_sfc = np.ndarray([0, 6], dtype=np.int32)  # [[B_,1,4,3,2,0]]
        self.c = []
        self.d_sum = 0.0

        self.counter = 0.0

        self.vnf_id_collection = []
        
        # energy
        self.energy_OIA = 0
        self.norm_energy = 0

        self.total_qoe = 0.0

    def reset(self, use_sfc_requests=None):
        energy.reset()
        self.__init__()
        self.B = group_config.get_initialized_bandwidth()
        self.D = group_config.get_initialized_delay()

        # Energy of each type of VNFs
        self.E_vnf = group_config.get_initialized_vnf()
        
        # IDLE Energy of each type of VNFs
        self.E_IDLEvnf = group_config.get_initialized_IDLEvnf(energy.idle_energy)
        
        # ACTIVE, IDLE, OFF
        self.state = group_config.get_initialized_state()

        # Number of each type of VNFs on each node
        self.capaVNFs = group_config.get_initialized_capaVNFs()
        
        # IDLE time of each node
        self.idletime = group_config.get_initialized_idletime()
        
        self.vnf_id_collection = []

        self.energy_OIA = 0
        
        self.S = None
        if use_sfc_requests:
            self.sfc_requests = use_sfc_requests
        else:
            self.sfc_requests = [[np.random.randint(self.B_min, self.B_max+1),
                                  np.random.randint(self.D_min, self.D_max+1)] for _ in range(self.num_requests)]
            #self.sfc_requests_energy = [np.random.randint(self.E_min, self.E_max+1) for _ in range(self.num_requests)]

    def start(self, B_, D_):
        self.B_ = B_
        self.D_ = D_
        self.c = []
        self.d_sum = 0.0
        self.random_release_sfc()
        self.S = np.concatenate([self.B,                        # 0-3
                                 np.ones([5, 5, 1])*self.B_,    # 4
                                 self.D,                        # 5-8
                                 np.ones([5, 5, 1])*self.D_,    # 9
                                 np.ones([5, 5, 1])*self.d_sum, # 10
                                 np.ones((5, 5, 1)),            # 11
                                 np.zeros([5, 5, 1]),           # 12
                                 np.zeros([5, 5, 1]),           # 13
                                 np.zeros([5, 5, 1]),           # 14
                                 np.zeros([5, 5, 1])], axis=2)  # 15
        return self.S

    def random_release_sfc(self, thresh=0.2):
        prob = np.array([np.random.random() for _ in range(self.running_sfc.shape[0])])
        released_sfc = self.running_sfc[prob <= thresh]
        self.running_sfc = self.running_sfc[prob > thresh]
        for c in released_sfc:
            for i in range(4):
                self.B[c[i+2], c[i+1], i] += c[0]
                

    def allocate_bandwidth(self, c, B_):
        for i in range(4):
            self.B[c[i+1], c[i], i] -= B_

    def step(self, action):
        vnf_id = np.argmax([self.S[..., 11][0, 0],
                            self.S[..., 12][0, 0],
                            self.S[..., 13][0, 0],
                            self.S[..., 14][0, 0],
                            self.S[..., 15][0, 0]])
        #print("vnf_id: ", vnf_id)
        if vnf_id == 0:
            self.vnf_id_collection = []
            self.c.append(action)
            print("Placement: ", self.c)
            self.S = np.concatenate([self.B,                        # 0-3
                                     np.ones([5, 5, 1])*self.B_,    # 4
                                     self.D,                        # 5-8
                                     np.ones([5, 5, 1])*self.D_,    # 9
                                     np.ones([5, 5, 1])*self.d_sum, # 10
                                     np.zeros((5, 5, 1)),           # 11
                                     np.ones([5, 5, 1]),            # 12
                                     np.zeros([5, 5, 1]),           # 13
                                     np.zeros([5, 5, 1]),           # 14
                                     np.zeros([5, 5, 1])], axis=2)  # 15
            reward = 0
            done = False
            self.vnf_id_collection.append(vnf_id)
            info = {'id': 0, 'msg': 'SUCCESS: Choose node {} from VNF{}.'.format(action, vnf_id)}
            print("INFO: ", info)
            return self.S, reward, done, info
        elif vnf_id in (1, 2, 3):
            if self.B[action, self.c[-1], vnf_id-1] < self.B_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                self.vnf_id_collection = []
                info = {'id': 1, 'msg': 'FAIL: Bandwidth not enough.'}
                print("INFO: ", info, "\n")
                self.total_qoe += reward
                return self.S, reward, done, info

            self.d_sum += self.D[action, self.c[-1], vnf_id-1]
            if self.d_sum > self.D_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                self.vnf_id_collection = []
                info = {'id': 2, 'msg': 'FAIL: Delay over constraint'}
                print("INFO: ", info, "\n")
                self.total_qoe += reward
                return self.S, reward, done, info

            self.c.append(action)
            print("Placement: ", self.c)
            self.S = np.concatenate([self.B,                        # 0-3
                                     np.ones([5, 5, 1])*self.B_,    # 4
                                     self.D,                        # 5-8
                                     np.ones([5, 5, 1])*self.D_,    # 9
                                     np.ones([5, 5, 1])*self.d_sum, # 10
                                     np.zeros((5, 5, 1)),           # 11
                                     np.zeros([5, 5, 1]),           # 12
                                     np.zeros([5, 5, 1]),           # 13
                                     np.zeros([5, 5, 1]),           # 14
                                     np.zeros([5, 5, 1])], axis=2)  # 15
            self.S[..., 12+vnf_id] = np.ones([5, 5])
            reward = 0
            done = False
            self.vnf_id_collection.append(vnf_id)
            info = {'id': 0, 'msg': 'SUCCESS: Choose node {} from VNF{}.'.format(action, vnf_id)}
            print("INFO: ", info)
            return self.S, reward, done, info
        else:
            if self.B[action, self.c[-1], vnf_id-1] < self.B_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                self.vnf_id_collection = []
                info = {'id': 1, 'msg': 'FAIL: Bandwidth not enough.'}
                print("INFO: ", info, "\n")
                self.total_qoe += reward
                return self.S, reward, done, info

            self.d_sum += self.D[action, self.c[-1], vnf_id-1]
            if self.d_sum > self.D_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                self.vnf_id_collection = []
                info = {'id': 2, 'msg': 'FAIL: Delay over constraint'}
                print("INFO: ", info, "\n")
                self.total_qoe += reward
                return self.S, reward, done, info

            self.c.append(action)
            print("Placement: ", self.c)
                
            # 添加到待釋放SFC列表
            sfc = [self.B_]
            sfc += self.c
            self.running_sfc = np.concatenate([self.running_sfc, np.array([sfc])], axis=0)
            # 分配頻寬資源
            self.allocate_bandwidth(self.c, self.B_)

            
            self.S = np.concatenate([self.B,                        # 0-3
                                     np.ones([5, 5, 1])*self.B_,    # 4
                                     self.D,                        # 5-8
                                     np.ones([5, 5, 1])*self.D_,    # 9
                                     np.ones([5, 5, 1])*self.d_sum, # 10
                                     np.zeros((5, 5, 1)),           # 11
                                     np.zeros([5, 5, 1]),           # 12
                                     np.zeros([5, 5, 1]),           # 13
                                     np.zeros([5, 5, 1]),           # 14
                                     np.zeros([5, 5, 1])], axis=2)  # 15
            
            self.vnf_id_collection.append(vnf_id)
            info = {'id': 0, 'msg': 'SUCCESS: Choose node {} from VNF{}. Complete a SFC request.'.format(action, vnf_id)}
            print("INFO: ", info)
            
            # Energy 
            # ===================================================================================================
            print("===========================================")
            # Energy evaluation
            print("VNF No.:           ", self.vnf_id_collection)
            print("Placement of VNFs: ", self.c)
            energy.update(self.c, self.state, self.idletime, self.vnf_id_collection, self.capaVNFs)
            print("State:                   ", self.state)
            #print("Energy of each VNF:      ", self.E_vnf)
            #print("IDLE Energy of each VNF: ", self.E_IDLEvnf)
            print("\nAllocate_of_eachVNF:")
            print("VNF:")
            print("  0 1 2 3 4")
            print(self.capaVNFs)
            print("IDLE time countdown: ", self.idletime)
            self.energy_OIA = energy.energy_OIA_mode(self.c, self.E_vnf, self.E_IDLEvnf,
                                                     self.state, self.vnf_id_collection, self.capaVNFs)
            print("Energy of OIA: ", round(self.energy_OIA, 0))
            print("===========================================\n")
            
            self.norm_energy = (self.energy_OIA-self.E_min)/(self.E_max-self.E_min)
            # ===================================================================================================
            
            # Test
            #with open('log_exp.txt', 'a') as f:
            #    f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}\n'.format(
            #        self.B_, self.D_, self.d_sum, self.norm_energy,
            #        np.log(self.B_), np.exp(-(self.D_-self.d_sum)), np.exp(-self.norm_energy))
            #    )
            
            reward = np.log(self.B_) - P*(np.exp(-(self.D_-self.d_sum)/10.0)+ np.exp(-self.energy_OIA))
            
            qoe = np.log(self.B_) - P*np.exp(-(self.D_-self.d_sum)/10.0)
            done = True
            self.total_qoe += qoe

            return self.S, reward, done, info

    def get_mean_qoe(self):
        return self.total_qoe / self.num_requests

if __name__ == '__main__':
    env = VNFGroup()
    env.reset()
    print(env.sfc_requests)
    #print(env.B[2,4,:])

    #tt = [4,3,2,1,0]
    #for item in tt:
    #    env.flag[item] = 2
    #    env.capacity[item] -= 1
    #print(env.flag)
    #print(env.capacity)
    