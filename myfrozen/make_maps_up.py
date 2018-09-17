import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

import numpy as np
import random

mydict = {'-10.0':'S', '-1.0':'H', '0.0':'F', '10.0':'G'}

def make_map(seed):
    
    STATEGRID = np.zeros((8,8))
    
    random.seed(seed)
    
    random_goal = (0,random.randint(0,7))
    random_start = (7,random.randint(0,7))

    STATEGRID[random_start] = -10
    STATEGRID[random_goal] = 10

    for row in range(1,7):
        random_col = random.randint(0,7)
        STATEGRID[row,random_col] = -1
    
    mymap = {}
    mymap['8x8'] = list()
    for number_row in range(0,8):
        rowletter = [mydict[element] for element in map(str, STATEGRID[number_row])]
        rowletter = ''.join(rowletter)
        mymap['8x8'].append(rowletter)
        
    return mymap