from collectMethodVec import saveAllDataToRam as saveAllDataToRamFirst
from my_main_first_2one2_1 import data_pre_deal as data_pre_dealFirst
from collectMethodVecAfter2one import saveAllDataToRam as saveAllDataToRamAfter
from my_main_after_2one import data_pre_deal as data_pre_dealAfter
from classification.utils import firstPathArrayCFG,firstPathArrayPDG,afterPathArrayCFG,afterPathArrayPDG
import torch
import torch.nn as nn
import torch.optim as optim
from MLP import Autoencoder, deal_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dealAST import getAst


# 定义所有的版本地址数组
# CFG
# firstPathArray = firstPathArrayCFG
# hidden = 32
# dtype = torch.float32
# PDG
firstPathArray = firstPathArrayPDG
hidden = 16
dtype = torch.float32

# first2one
def dealFirst():
    for path in firstPathArray:
        jsonVecPath = path[0]
        bugDataPath = path[1]
        outPath = path[2]

        ramData = saveAllDataToRamFirst(jsonVecPath, bugDataPath, hidden, dtype)
        data_pre_dealFirst(jsonVecPath, ramData, outPath)

dealFirst()

# CFG
# afterPathArray = afterPathArrayCFG

# PDG
afterPathArray = afterPathArrayPDG
# after2one
def dealAfter():
    for path in afterPathArray:
        jsonVecPath = path[0]
        bugDataPath = path[1]
        outPath = path[2]
        ramData = saveAllDataToRamAfter(jsonVecPath, bugDataPath, hidden, dtype)
        data_pre_dealAfter(jsonVecPath, ramData, outPath)


dealAfter()