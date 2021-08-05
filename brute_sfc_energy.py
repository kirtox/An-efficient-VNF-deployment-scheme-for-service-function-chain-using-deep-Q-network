# coding=utf-8
from config import VNFGroupConfig
from energy import Energy
from env import VNFGroup
import numpy as np
import time

group_config = VNFGroupConfig()
energy = Energy()
env = VNFGroup()

P = 10.0


class BruteSFC_Energy:
    def __init__(self):
        self.B = group_config.get_initialized_bandwidth()
        self.D = group_config.get_initialized_delay()
        self.sfc_requests = group_config.get_test_sfc()
        self.running_sfc = np.ndarray([0, 6], dtype=np.int32)
        self.total_qoe = 0.0
        self.error_counter = 0
        
        # ============================================================================================
        self.time = 0
        
        self.vnf_id_collection = [0, 1, 2, 3, 4]

        # Energy of each type of VNFs
        self.E_vnf = group_config.get_initialized_vnf()
        
        # IDLE Energy of each type of VNFs
        self.E_IDLEvnf = group_config.get_initialized_IDLEvnf(energy.idle_energy)
        
        # 0:OFF, 1:IDLE, 2:ACTIVE
        self.state = group_config.get_initialized_state()

        # Number of each type of VNFs on each node
        self.capaVNFs = group_config.get_initialized_capaVNFs()
        
        # IDLE time of each node
        self.idletime = group_config.get_initialized_idletime()

        self.energy_OIA = 0
        self.lowest_energy = 9999999
        
        self.total_OIA = []
        # ============================================================================================
        
    def reset(self):
        energy.reset()
        self.__init__()
        
    def set_sfc_requests(self, sfc_requests):
        self.sfc_requests = sfc_requests

    def random_release_sfc(self, thresh=0.2):
        prob = np.array([np.random.random() for _ in range(self.running_sfc.shape[0])])
        released_sfc = self.running_sfc[prob <= thresh]
        self.running_sfc = self.running_sfc[prob > thresh]
        for c in released_sfc:
            for i in range(0, 4):
                self.B[c[i+2], c[i+1], i] += c[0]

    def check_B(self, c, B_):
        for vnf_id in range(1, 5):
            if self.B[c[vnf_id], c[vnf_id-1], vnf_id-1] < B_:
                return False
        return True

    def check_D(self, c, D_):
        d_sum = 0.0
        for vnf_id in range(1, 5):
            d_sum += self.D[c[vnf_id], c[vnf_id-1], vnf_id-1]
        if d_sum > D_:
            return 0, False
        return d_sum, True

    def allocate_B(self, c, B_):
        for vnf_id in range(1, 5):
            self.B[c[vnf_id], c[vnf_id-1], vnf_id-1] -= B_

    def select(self):
        start = time.time()
        [B_, D_] = self.sfc_requests.pop(0)

        while True:
            self.random_release_sfc()
            
            self.lowest_energy = 9999999
            best_c = None
            best_qoe = 0.0
            for node1 in range(5):
                for node2 in range(5):
                    for node3 in range(5):
                        for node4 in range(5):
                            for node5 in range(5):
                                c = [node1, node2, node3, node4, node5]

                                if self.check_B(c, B_):
                                    d_sum, flag = self.check_D(c, D_)
                                    if flag:
                                        qoe = np.log(B_) - P*np.exp(-(D_-d_sum)/10.0)
                                        
                                        # backup
                                        # ===================================================================================================
                                        global tmp_state, tmp_capaVNFs, tmp_idletime
                                        tmp_state = np.copy(self.state)
                                        tmp_capaVNFs = np.copy(self.capaVNFs)
                                        tmp_idletime = np.copy(self.idletime)
                                        # ===================================================================================================
                                        
                                        # Energy 
                                        # ===================================================================================================
                                        #print("===========================================")
                                        #print("VNF No.:           ", self.vnf_id_collection)
                                        #print("Placement of VNFs: ", c)
                                        energy.update(c, self.state, self.idletime, self.vnf_id_collection, self.capaVNFs)
                                        #print("State:                   ", self.state)
                                        #print("Energy of each VNF:      ", self.E_vnf)
                                        #print("IDLE Energy of each VNF: ", self.E_IDLEvnf)
                                        #print("\nAllocate_of_eachVNF:")
                                        #print("VNF:")
                                        #print("  0 1 2 3 4")
                                        #print(self.capaVNFs)
                                        #print("IDLE time countdown: ", self.idletime)
                                        self.energy_OIA = energy.energy_OIA_mode(c, self.E_vnf, self.E_IDLEvnf, 
                                                                                 self.state, self.vnf_id_collection, self.capaVNFs)
                                        #print("Energy of OIA: ", round(self.energy_OIA, 0))
                                        #print("===========================================\n")
                                        # ===================================================================================================

                                        if self.energy_OIA < self.lowest_energy:
                                            #print(self.energy_OIA, "&&", self.lowest_energy)
                                            self.lowest_energy = self.energy_OIA
                                            best_qoe = qoe
                                            best_c = c

                                        else:
                                            # recovery
                                            energy.OIA_energy -= self.energy_OIA
                                            self.state = tmp_state
                                            self.capaVNFs = tmp_capaVNFs
                                            self.idletime = tmp_idletime                 
                                        
            #print('Period: {:g}'.format(round((end-start), 0)))
            if best_c:
                self.total_qoe += best_qoe
                self.allocate_B(best_c, B_)
                sfc = [B_]
                sfc += best_c
                self.running_sfc = np.concatenate([self.running_sfc, np.array([sfc])], axis=0)
                
                # lowest energy add
                self.total_OIA.append(self.lowest_energy)
                
                                        
            else:
                self.total_qoe -= P
                self.error_counter += 1
            try:
                [B_, D_] = self.sfc_requests.pop(0)
            except IndexError:
                break
        end = time.time()
        self.time = round((end-start), 2)

    def get_mean_qoe(self):
        return self.total_qoe / env.num_requests

    def get_error_rate(self):
        return self.error_counter / env.num_requests
    
    def get_mean_energy_OIA(self):
        return sum(self.total_OIA) / env.num_requests
    
    def get_max_energy_OIA(self):
        return max(self.total_OIA)


if __name__ == '__main__':
    sfc = BruteSFC_energy()
    sfc.select()
    
    print('Period: {:g}s '.format(sfc.time))

    print('Mean QoE:', sfc.get_mean_qoe())
    print('Error Rate:', sfc.get_error_rate())
    print("Mean Energy of OIA: ", round(sfc.get_mean_energy_OIA(), 0))
    print("Max Energy of OIA: ", round(sfc.get_max_energy_OIA(), 0))
    tt = sfc.total_OIA

    