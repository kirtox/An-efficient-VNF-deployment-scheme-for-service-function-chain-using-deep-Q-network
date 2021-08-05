import numpy as np


class VNFGroupConfig:
    def get_initialized_bandwidth(self):
        return np.array(
            [[[1261, 1035, 1074, 1243],
              [1243, 1246, 832, 1223],
              [1041, 1011, 1181, 891],
              [1272, 1074, 1119, 1044],
              [1105, 1069, 908, 1181]],

             [[1114, 925, 958, 1013],
              [1099, 824, 895, 1135],
              [1076, 1256, 768, 801],
              [1280, 910, 1140, 856],
              [1187, 1054, 774, 1222]],

             [[1090, 1136, 1208, 889],
              [823, 1130, 1187, 1254],
              [1182, 1257, 1183, 964],
              [1164, 904, 1090, 1179],
              [866, 982, 976, 1219]],

             [[1078, 1178, 1104, 789],
              [1188, 1039, 867, 970],
              [907, 1065, 1142, 870],
              [1071, 841, 972, 1052],
              [931, 1176, 986, 863]],

             [[890, 920, 858, 836],
              [799, 867, 1278, 910],
              [948, 842, 1037, 1146],
              [827, 833, 1126, 843],
              [1102, 1041, 1069, 814]]])

    def get_initialized_delay(self):
        return np.array(
            [[[18, 10, 18, 16],
              [10, 14, 15, 17],
              [11, 10, 14, 15],
              [16, 17, 14, 14],
              [14, 14, 12, 10]],

             [[20, 10, 15, 14],
              [20, 11, 14, 19],
              [13, 18, 12, 19],
              [18, 14, 10, 14],
              [17, 11, 16, 18]],

             [[19, 17, 10, 19],
              [18, 10, 14, 13],
              [20, 10, 12, 19],
              [17, 12, 15, 20],
              [11, 11, 16, 11]],

             [[15, 13, 18, 16],
              [13, 11, 20, 12],
              [12, 13, 11, 17],
              [13, 11, 11, 12],
              [18, 16, 16, 13]],

             [[15, 15, 17, 16],
              [15, 13, 19, 17],
              [17, 10, 10, 13],
              [17, 15, 18, 13],
              [19, 19, 12, 10]]])


    # energy consumption of each VNF
    """
    def get_initialized_vnf(self):
        return np.array(
            np.random.randint(15,21,size=5))
    """
    def get_initialized_vnf(self):
        return np.array(
            [16, 20, 17, 15, 17])
    
    
    # IDLE energy consumption of each VNF 
    #ex: vnf0: ACTIVE->16, so IDLE = (16/340)*(340*60%) = 4.8(J)
    """
    def get_initialized_IDLEvnf(self, vnf, idle_energy):
        return np.array(
            vnf*idle_energy)
    """
    def get_initialized_IDLEvnf(self, idle_energy):
        return np.array(
            [16*idle_energy, 20*idle_energy, 17*idle_energy, 15*idle_energy, 17*idle_energy])
    
    
    # flag of each node (ACTIVE: 2, IDLE: 1, OFF: 0)
    def get_initialized_state(self):
        return np.array(
            ['OFF', 'OFF', 'OFF', 'OFF', 'OFF'], dtype='<U6')
    
    # max numbers of each VNFs, each VNF have 20
    def get_initialized_capaVNFs(self):
        return np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
            )
    
    # max idletime of each VNF on each node
    # ACTIVE: 1, IDLE: -3, -2, -1, OFF: 0
    def get_initialized_idletime(self):
        return np.array(
            [0, 0, 0, 0, 0])
    """
    def get_initialized_runtime(self):
        return np.array(
            [0, 0, 0, 0, 0])
    """
    def get_test_sfc(self):
        return [[53, 62], [88, 59], [219, 55], [95, 50], [160, 51], [220, 57], [253, 56], [41, 70], [117, 68],
                [228, 61], [250, 78], [173, 64], [66, 54], [231, 73], [249, 80], [112, 72], [157, 59], [23, 72],
                [73, 51], [144, 67], [24, 74], [157, 58], [46, 57], [147, 56], [165, 53], [212, 74], [59, 62], [42, 66],
                [125, 68], [31, 50], [212, 75], [127, 84], [231, 57], [42, 75], [120, 72], [25, 53], [247, 73],
                [141, 86], [171, 87], [73, 69], [182, 58], [48, 84], [26, 73], [159, 73], [41, 57], [108, 60],
                [206, 82], [232, 73], [199, 51], [129, 63], [19, 50], [221, 56], [68, 71], [214, 52], [92, 77],
                [165, 61], [23, 63], [216, 61], [92, 70], [46, 86], [119, 57], [61, 54], [73, 68], [112, 63], [26, 73],
                [140, 67], [151, 74], [218, 78], [164, 82], [156, 51], [110, 74], [98, 83], [146, 78], [119, 84],
                [26, 82], [249, 68], [230, 56], [210, 57], [247, 65], [149, 67], [186, 70], [134, 72], [245, 63],
                [130, 83], [197, 70], [26, 82], [199, 75], [165, 89], [112, 69], [41, 64], [157, 70], [132, 56],
                [221, 65], [40, 59], [210, 57], [69, 55], [180, 87], [37, 90], [93, 77], [193, 63]]
