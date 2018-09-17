import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

import numpy as np
import random

mydict = {'-10.0':'S', '-1.0':'H', '0.0':'F', '10.0':'G'}

def make_map(seed,rowindex,colindex):
    
    STATEGRID = np.zeros((8,8))
    
    random.seed(seed)
    
    #random_goal = (rowindex,colindex)
    #random_start = (random.randint(0,7),0)

    #STATEGRID[random_start] = -10
    #STATEGRID[random_goal] = 10
    rowstart=0
    colstart=7
    for col in range(0,7):
        random_row = random.randint(0,7)
        STATEGRID[random_row,col] = -1
    
    if colindex==0 or colindex==7:
        if colindex==0:
            colstart=7
        elif colindex==7:
            colstart=0
        random_start = (random.randint(0,7),colstart)
    else:
        if rowindex==0:
            rowstart=7
        elif rowindex==7:
            rowstart=0
        random_start = (rowstart,random.randint(0,7))
    
    random_goal = (rowindex,colindex)
    #random_start = (random.randint(0,7),0)

    STATEGRID[random_start] = -10
    STATEGRID[random_goal] = 10
    
    mymap = {}
    mymap['8x8'] = list()
    for number_row in range(0,8):
        rowletter = [mydict[element] for element in map(str, STATEGRID[number_row])]
        rowletter = ''.join(rowletter)
        mymap['8x8'].append(rowletter)
        
    return mymap