import math
import numpy as np

# define global variable transition_probs
transition_probs = {}
states = []

def get_state_format(state1, state2):
    if len(state1)
    input("state format not found. state:", state)


def get_transition_prob(state1, state2):
    # can be further optimized if functionality is correct (by storing pre-defined lists or hashmaps)
    # if same letter/triletter sequence, return transition from provided table
    # if different letters:
    #   if state1[-1] between 0 and 4 inclusive, return 0
    #   if state2[-1] between 1 and 5 inclusive, return 0
    #   if state1[-1] is 5 and state2[-1] is 0, calculate probability and return
    #       if format1 is "a-b"
    #           return 0
    #       if format1 is "a+b"
    #           if format2 is not "a-b+c" or "a-b":
    #               return 0
    #           else:
    #               calculate total number of "a-b+c" and "a-b" options for all c
    #               return 1/calculated_number
    #       if format1 is "a-b+c":
    #           if format2 is not "b-c+d" or "b-c":
    #               return 0
    #           else:
    #               calculate total number of "b-c+d" and "b-c" options for all c
    #               return 1/calculated_number
    ## for later, check number of states in sil0, sil1 and change above if statements
    state_num1 = int(state1[-1])
    state_num2 = int(state2[-1])
    if state_num1 == state_num2:
        return transition_probs[state1][state2]
    else:
        if 0 <= state_num1 <= 4 or 1 <= state_num2 <= 5:
            return 0
        elif state_num1 == 5 and state_num2 == 0:
            if len(state1) == 3 and state1[1] == "-":
                return 0
            if len(state1) == 3 and state1[1] == "+":
                if not (len(state2) == 5 and state2[1] == "-" and state2[3] == "+" and state2[0] == state1[0] and state2[2] == state1[2]) and \
                    not (len(state2) == 3 and state2[1] == "-" and state2[0] == state1[0] and state2[2] == state1[2]):
                    return 0
                else:
                    count = 0
                    for s in states:
                        if (len(s) == 5 and s[1] == "-" and s[3] == "+" and s[0] == state1[0] and s[2] == state1[2]) or \
                            (len(s) == 3 and s[1] == "-" and s[0] == state1[0] and s[2] == state1[2]):
                            count += 1
                    return 1/count
            elif len(state1) == 5 and state1[1] == "-" and state1[3] == "+":
                pass



    input(f"transition_prob not found for state 1 '{state1}' and state2 '{state2}")


def gaussian_prob(x, para_tuple):

    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    std = math.sqrt(std)
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile

def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras, ndim=2):
    sequence = []
    probability = 0.0

    if len(evidence_vector) == 0:
        return sequence, probability

    nl = []

    for i in range(len(states)):
        prod = np.log(1)
        for z in range(ndim):
            # prod = prod * gaussian_prob(evidence_vector[0][z], emission_paras[states[i]][z])
            prod = prod + np.log(gaussian_prob(evidence_vector[0][z], emission_paras[states[i]][z]))
        # nl.append([prior_probs[states[i]] * prod] + [0] * (len(evidence_vector)-1))
        nl.append([np.log(prior_probs[states[i]]) + prod] + [0] * (len(evidence_vector) - 1))

    for i in range(1, len(evidence_vector)):
        for j in range(len(states)):
            state = states[j]
            if j >= 1 and i >= 1:
                max_val = -math.inf
                best_prev_prob = None
                k_new = None
                for k in range(len(states)):
                    # if states[j] in transition_probs[states[k]] and nl[k][i-1]*transition_probs[states[k]][states[j]] > max_val:
                    if states[j] in transition_probs[states[k]] and (nl[k][i-1] + np.log(transition_probs[states[k]][states[j]])) >= max_val:
                        # max_val = nl[k][i-1] * transition_probs[states[k]][states[j]]
                        max_val = nl[k][i - 1] + np.log(transition_probs[states[k]][states[j]])
                        best_prev_prob = nl[k][i-1]
                        k_new = k
                prev_prob = best_prev_prob
                prev_state = states[k_new]
            elif i >= 1:
                prev_prob = nl[j][i - 1]
                prev_state = states[j]
            a = np.log(1)
            for z in range(ndim):
                # a = a * gaussian_prob(evidence_vector[i][z], emission_paras[state][z])
                a = a + np.log(gaussian_prob(evidence_vector[i][z], emission_paras[state][z]))
            # nl[j][i] = prev_prob * a * transition_probs[prev_state][state]
            nl[j][i] = prev_prob + a + np.log(transition_probs[prev_state][state])

    new_s = []
    seq = []
    highest_prob = -math.inf
    highest_prob_index = None
    for j in range(len(states)):
        if highest_prob <= nl[j][-1]:
            # if (states[j] in ['sil03', 'sil13', '_2'] or states[j][-1] == '7') and highest_prob <= nl[j][-1]:
            if highest_prob <= nl[j][-1]:

                highest_prob = nl[j][-1]
                highest_prob_index = j
    new_s.append(highest_prob)
    sequence.append(states[highest_prob_index])
    probability = highest_prob
    for i in range(len(evidence_vector)-2,-1,-1):
        change_j = None
        highest_prob = -math.inf
        new_highest_prob = -math.inf
        best_state = None
        nj = None
        ni = None

        for j in range(len(states)):
            if sequence[0] not in transition_probs[states[j]]:
                continue
            # if nl[j][i]*transition_probs[states[j]][sequence[0]] > highest_prob:
            if (nl[j][i] + np.log(transition_probs[states[j]][sequence[0]])) > highest_prob:
                # highest_prob = nl[j][i]*transition_probs[states[j]][sequence[0]]
                highest_prob = nl[j][i] + np.log(transition_probs[states[j]][sequence[0]])
                new_highest_prob = nl[j][i]
                best_state = states[j]
                change_j = j
                nj = j
                ni = i
        if best_state:
            sequence = [best_state] + sequence
            new_s = [new_highest_prob] + new_s
            seq = seq + [(nj, ni)]
            

    if probability == 0:
        return (None, 0)

    return sequence, probability



if __name__ == "__main__":
    states_ = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2']
    prior_probs_ = {}
    for state in states_:
        if state[-1] == '1':
            prior_probs_[state] = 1/3
        else:
            prior_probs_[state] = 0
    transition_probs_ = {}
    for state in states_:
        transition_probs_[state] = {}
        for state1 in states_:
            if state1[0] == state[0]:
                transition_probs_[state][state1] = 0.33
            else:
                transition_probs_[state][state1] = 0
    emission_paras_ = {
        'a1': [(1.0,1.0), (1.0,1.0), (1.0,1.0)],
        'a2': [(1.0,1.0), (1.0,1.0), (1.0,1.0)],
        'a3': [(1.0,1), (1.0,1), (1.0,1)],
        'b1': [(1.0,1), (1.0,1), (1.0,1)],
        'b2': [(1.0,1), (1.0,1), (1.0,1)],
        'b3': [(1.0,1), (1.0,1), (1.0,1)],
        'c1': [(5.0,1), (1.0,1), (3.0,1)],
        'c2': [(10.0,1), (1.0,1), (3.0,1)]  
    }

    vector = [(5.0,1,3), (5.0,1,3), (5,1,3), (10,1,3), (10,1,3), (10,1,3), (10,1,3)]

    prior_probs_tensor = {'a1':0.33, 'a2':0, 'a3':0, 'b1':0.33, 'b2':0, 'b3':0, 'c1':0.33, 'c2':0}
    r = multidimensional_viterbi(evidence_vector=vector,states=states_, prior_probs=prior_probs_tensor, transition_probs=transition_probs_, emission_paras=emission_paras_)

    print(r)