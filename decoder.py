import math
import numpy as np


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


