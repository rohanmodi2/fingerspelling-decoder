import tensorflow as tf
import math

class HMMModel(tf.keras.Model):
    def __init__(self, states, prior_probs, transition_probs, emission_paras):
        super(HMMModel, self).__init__()
        self.states = states
        self.prior_probs = prior_probs
        self.transition_probs = transition_probs
        self.emission_paras = emission_paras

    def call(self, evidence_vector):
        print("evidence vector:")
        print(evidence_vector)
        return self.multidimensional_viterbi(evidence_vector,
                                             self.states,
                                             self.prior_probs,
                                             self.transition_probs,
                                             self.emission_paras)

    def gaussian_prob(self, x, para_tuple):
        if para_tuple[0] is None or para_tuple[1] is None:
            return tf.constant(0.0)
        mean, std = para_tuple
        gaussian_percentile = (2 * math.pi * std**2)**-0.5 * \
                            tf.math.exp(-(x - mean)**2 / (2 * std**2))
        return gaussian_percentile

    def multidimensional_viterbi(self, evidence_vector, states, prior_probs,
                                 transition_probs, emission_paras, ndim=2):
        sequence = []
        probability = 0.0

        num_states = len(states)
        num_evidence = len(evidence_vector)

        # Convert lists to tensors
        prior_probs = tf.constant(prior_probs, dtype=tf.float32)
        transition_probs = {state: tf.constant(transition_probs[state], dtype=tf.float32) for state in states}
        emission_paras = {state: tf.constant(emission_paras[state], dtype=tf.float32) for state in states}

        evidence_vector = tf.constant(evidence_vector, dtype=tf.float32)
        
        nl = []

        for i in range(num_states):
            prod = tf.constant(0.0)
            for z in range(ndim):
                prod += tf.math.log(self.gaussian_prob(evidence_vector[0][z], emission_paras[states[i]][z]))
            nl.append([tf.math.log(prior_probs[states[i]]) + prod] + [tf.constant(0.0)] * (num_evidence - 1))

        for i in range(1, num_evidence):
            for j in range(num_states):
                state = states[j]
                if j >= 1 and i >= 1:
                    max_val = -math.inf
                    best_prev_prob = None
                    k_new = None
                    for k in range(num_states):
                        prev_state = states[k]
                        transition_prob = transition_probs[prev_state].get(state, tf.constant(0.0))
                        prob = nl[k][i - 1] + tf.math.log(transition_prob)
                        if prob >= max_val:
                            max_val = prob
                            best_prev_prob = nl[k][i - 1]
                            k_new = k
                    prev_prob = best_prev_prob
                    prev_state = states[k_new]
                elif i >= 1:
                    prev_prob = nl[j][i - 1]
                    prev_state = states[j]
                else:
                    prev_prob = tf.constant(0.0)
                    prev_state = states[j]
                    
                a = tf.constant(0.0)
                for z in range(ndim):
                    a += tf.math.log(self.gaussian_prob(evidence_vector[i][z], emission_paras[state][z]))
                
                nl[j][i] = prev_prob + a + tf.math.log(transition_probs[prev_state][state])

        new_s = []
        seq = []

        highest_prob = -math.inf
        highest_prob_index = None
        for j in range(num_states):
            if highest_prob <= nl[j][-1]:
                highest_prob = nl[j][-1]
                highest_prob_index = j

        new_s.append(highest_prob)
        sequence.append(states[highest_prob_index])
        probability = highest_prob

        for i in range(num_evidence - 2, -1, -1):
            change_j = None
            highest_prob = -math.inf
            new_highest_prob = -math.inf
            best_state = None
            nj = None
            ni = None

            for j in range(num_states):
                if sequence[0] not in transition_probs[states[j]]:
                    continue
                prob = nl[j][i] + tf.math.log(transition_probs[states[j]][sequence[0]])
                if prob > highest_prob:
                    highest_prob = prob
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
            return None, 0

        return sequence, probability
