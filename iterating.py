import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
from IPython.display import clear_output
import time
import numpy as np
import pandas as pd
from sklearn import linear_model
import math
import gym
import myfrozen
import pickle
from gym.envs.registration import register, spec

def run_episode(env,Q,learning_rate,gamma,episode,zerostates):
    observation = env.reset()
    done = False
    t_reward = 0
    max_steps = 100

    move_counter = 0
    for j in range(max_steps):
        if done:
            if j < max_steps:
                zerostates.add(observation)
            break

        curr_state = observation

        #action = np.argmax(Q[curr_state,:]  + np.random.randn(1, env.action_space.n)*(1./(episode+1)))
        action = np.argmax(Q[curr_state,:]  + np.random.randn(1, env.action_space.n))
        
        move_counter+=1
        
        observation, reward, done, info = env.step(action)
        
        collect.append((curr_state,action,observation,reward))

        t_reward += reward

        #Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        Q[curr_state,action] += learning_rate * (reward+ gamma*np.max(Q[observation,:])-Q[curr_state,action])

    return Q, t_reward, done, move_counter

def trainer(iteration,epochs=1000,learning_rate = 0.81,discount = 0.96):
    
    reward_per_ep = list()
    wins = 0
    losses = 0
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for i in range(epochs):
        #print i
        Q, t_reward, done, move_counter = run_episode(env,Q,learning_rate,discount,i,zerostates)
        #print done
        reward_per_ep.append(t_reward)

        if done:
            if t_reward > 0 : # Win
                wins += 1
            else: # Loss
                losses += 1
        time.sleep(.1)        
        clear_output(wait=True)
        print("Board #: %s" % (iteration+1,))
        print("Game #: %s" % (i+1,))
        print("Moves this round %s" % move_counter)
        print("Final Position:")
        env.render()
        print("Wins/Losses %s/%s" % (wins, losses))

    return Q, reward_per_ep, collect, zerostates

def make_model(collect):
    df = pd.DataFrame(data=collect, columns=['state','action','state_new','reward'])

    one_hot = pd.get_dummies(df['state'],prefix='state')
    df = df.drop('state',axis=1)
    df = df.join(one_hot)
    one_hot = pd.get_dummies(df['state_new'],prefix='new_state')
    df = df.drop('state_new',axis=1)
    df = df.join(one_hot)
    one_hot = pd.get_dummies(df['action'],prefix='action')
    df = df.drop('action',axis=1)
    df = df.join(one_hot)

    target = pd.DataFrame(df.reward, columns=["reward"])
    df = df.drop('reward',axis=1)

    lm = linear_model.LinearRegression()
    model = lm.fit(df,target['reward'])

    return model

def to_onehot(size,value):
  my_onehot = np.zeros((size))
  my_onehot[value] = 1.0
  return my_onehot

def make_environment(name,seed):
    register(
            id=name,
            entry_point='myfrozen.frozen_lake:FrozenLakeEnv',
            kwargs={'map_name': '8x8', 'is_slippery': False},
            timestep_limit=100,
            reward_threshold=0.78,
            )
    env = gym.make(name)
    return env

def fill_dict(D,MY_ENV_NAME,env,q,zerostates,policy,grid,model):
    D[MY_ENV_NAME]["env"] = env
    D[MY_ENV_NAME]["q"] = q
    D[MY_ENV_NAME]["zerostates"] = zerostates
    D[MY_ENV_NAME]["policy"] = policy
    D[MY_ENV_NAME]["grid"] = grid 
    D[MY_ENV_NAME]["model"] = model
    return D

def make_policy(OBSERVATION_SPACE,q):
    policy = np.zeros(OBSERVATION_SPACE)
    for i in range(OBSERVATION_SPACE):
        policy[i] = np.argmax(q[i,:])
    return policy

def show_policy(initial_state):
    A2A=['<','v','>','^']
    grid = np.zeros((OBS_SQR,OBS_SQR), dtype='<U2')
    for x in range(0,OBS_SQR):
        for y in range(0,OBS_SQR):
            my_state = initial_state.copy()
            my_state[x,y] = 1

            obs_predict = my_state.reshape(1,OBSERVATION_SPACE,)
            obs_predict = np.squeeze(obs_predict)
            index, = np.where(obs_predict == 1.)
            action = np.argmax(q[index,:])
            grid[x,y] = A2A[action]
    grid
    return grid

D = {}
n_iters = 100
for iteration in xrange(n_iters):
    name='FrozenLakeNonskid8x8-v%d' % iteration
    env = make_environment(name,iteration)
    
    OBSERVATION_SPACE = env.observation_space.n
    ACTION_SPACE = env.action_space.n
    
    collect = []
    zerostates = set()
    
    q, rpe, collect, zerostates = trainer(iteration,epochs=2000)
    collect = np.array(collect)
    policy = make_policy(OBSERVATION_SPACE,q)
    model = make_model(collect)
    
    OBS_SQR = int(math.sqrt(OBSERVATION_SPACE))
    STATEGRID = np.zeros((OBS_SQR,OBS_SQR))
    
    grid = show_policy(STATEGRID)
    
    D[name] = {}
    D = fill_dict(D,name,env,q,zerostates,policy,grid,model)
    
    env.close()
    del name,env,q,zerostates,policy,model

print("SAVING RESULTS")
output = open('saved/collected2.pkl', 'wb')
pickle.dump(D, output)
output.close()
print("DONE")