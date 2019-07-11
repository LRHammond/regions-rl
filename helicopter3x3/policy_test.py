# Code for comparing a learnt policy across regions

import numpy as np
import helicopter3x3
import regions_tree_3x3
from collections import defaultdict
from utils import encode_state
import os
import sys
import tensorflow as tf
import copy

# Generates all of the 512 possible maps
def generate_all_maps():

    r = np.arange(2**9)
    r = r.astype(np.uint32).view(np.uint8)
    r = np.unpackbits(r).reshape((-1,32))[:,:9]
    r = np.argwhere(r.astype(bool))
    data = [defaultdict(lambda:[]) for _ in range(2**9)]
    for i in range(r.shape[0]):
        n , ix = r[i]
        p = helicopter3x3.Position(*np.unravel_index(ix,(3,3)))
        data[n][p] = True

    return data

# Creates all possible states with helicopter at position (x,y) and with fuel f
def generate_all_from(x,y,f):

    states = []
    for nmap in generate_all_maps():
        state = helicopter3x3.State()
        pos = helicopter3x3.Position(x,y)
        state.receive_SetState(pos, nmap, f)
        states.append(state)

    return states

# Generates all states within the game (including some that may not actually be reachable)
def generate_all():

    all_states = []
    for x in range(3):
        for y in range(3):
            all_states += generate_all_from(x,y,1)
    
    return all_states

# Outputs the region of a state*action*action triple
def region_from_state(state, m1, m2):

    nstate = copy.deepcopy(state)
    if nstate.receive_Move(m1) != None:
        return None
    if nstate.receive_Move(m2) != None:
        return None

    inputs = \
        [ type("",(),{"nmap":state.islands, "pos":state.position, "nfuel": state.fuel})
        , type("",(),{"pos":m1})
        , type("",(),{"pos":m2})
        , None
        ]
    
    region = regions_tree_3x3.calculate_regions(inputs)
    if region != None:
        region = tuple(region)

    return region

# Takes a learnt DQN as input and outputs the policy distribution over each region
def policy_test(nn, states, inputs):

    policy = {}
    predictions = nn.predict(inputs).reshape((-1,9))

    for i in range(512*3*3):

        choice = np.argmax(predictions[i])
        state = states[i]

        for a1 in range(9):
            for a2 in range(9):

                m1 = np.unravel_index(a1, (3,3))
                m1 = helicopter3x3.Position(*m1)
                m2 = np.unravel_index(a2, (3,3))
                m2 = helicopter3x3.Position(*m2)

                region = region_from_state(state, m1, m2)

                if region != None:

                    if region not in policy.keys():  
                        policy[region] = [0 for _ in range(9)]                      
                    policy[region][choice] += 1

    for region in policy.keys():
        policy[region] = [a / sum(policy[region]) for a in policy[region]]

    return policy

# Main function runs the test using a stored NN and outputs a CSV file
if __name__ == "__main__":

    r = sys.argv[1]
    n = int(sys.argv[2])

    states = generate_all()
    inputs = np.zeros((512*3*3,3,3,2))
    for s, state in enumerate(states):
        encode_state(inputs[s], state)

    for i in range(n):

        nn = tf.keras.models.load_model("nets/NN_{0}_{1}.h5".format(r,i), custom_objects={'tf': tf})
        policy = policy_test(nn, states, inputs)
        print("Policy {} {} evaluated!".format(r,i))

        with open("policy_dists/policy_{0}_{1}.csv".format(r,i), "w") as f:

            for region in policy.keys():
                f.write("".join([str(x) for x in region]) + ", " + ", ".join([str(y) for y in policy[region]]) + "\n")

    print("Done!")