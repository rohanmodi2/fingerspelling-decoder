import tensorflow as tf
import math
import numpy as np

class HMMModel(tf.keras.Model):
    def __init__(self, states, prior_probs, transition_probs, emission_paras):
        super(HMMModel, self).__init__()
        list_emission_paras = []
        for state in states:
            list_emission_paras.append(emission_paras[state])
        np_emission_paras = np.array(list_emission_paras)
        tensor_emission_paras = tf.convert_to_tensor(np_emission_paras, np.float32)

        tp_list = []
        for state_i in states:
            new_list = []
            for state_j in states:
                try:
                    new_list.append(transition_probs[state_i][state_j])
                except:
                    new_list.append(0)
            tp_list.append(new_list)
        np_tp = np.array(tp_list)
        tensor_tp = tf.convert_to_tensor(np_tp, np.float32)
        self.states = tf.range(len(states))
        self.prior_probs = prior_probs
        self.transition_probs = tensor_tp
        self.emission_paras = tensor_emission_paras

    @tf.function
    def call(self, evidence_vector):
        return self.multidimensional_viterbi(evidence_vector,
                                             self.states,
                                             self.prior_probs,
                                             self.transition_probs,
                                             self.emission_paras)
    @tf.function
    def gaussian_prob(self, x, para_tuple):
            if para_tuple[0] is None or para_tuple[1] is None:
                return tf.constant(0.0)
            mean, std = para_tuple
            gaussian_percentile = (2 * math.pi * std**2)**-0.5 * \
                                tf.math.exp(-(x - mean)**2 / (2 * std**2))
            return gaussian_percentile

    @tf.function
    def multidimensional_viterbi(self, evidence_vector, states, prior_probs,
                                transition_probs, emission_paras, ndim=2):
                sequence = tf.constant([], dtype=tf.int32)
                probability = 0.0

                # if tf.shape(evidence_vector)[0] == 0:
                #     return sequence, probability
                
                nl = tf.constant([])

                p = (2 * math.pi * emission_paras[:, :, 1]) ** (-0.5) * \
                tf.exp(-(evidence_vector[0] - emission_paras[:, :, 0])**2 / (2 * emission_paras[:, :, 1]**2))
                result = tf.reduce_sum(tf.math.log(p), axis=1, keepdims=True)

                tensor_column = tf.reshape(result, (tf.shape(result)[0], 1))
                zero_tensor = tf.zeros((tf.shape(result)[0], tf.shape(evidence_vector)[0] - 1), dtype=tf.float32)
                tensor_2d = tf.concat([tensor_column, zero_tensor], axis=1)
                nl = tensor_2d

                prev_prob = 0.0
                prev_state = tf.gather(states, 0)
                for i in range(1, tf.shape(evidence_vector)[0]):
                    for j in range(tf.shape(states)[0]):
                        state = tf.gather(states, j)
                        if j >= 1 and i >= 1:
                            max_val = -math.inf
                            best_prev_prob = 0.0
                            k_new = 0
                            for k in range(tf.shape(states)[0]):
                                # if states[j] in transition_probs[states[k]] and (nl[k][i-1] + np.log(transition_probs[states[k]][states[j]])) >= max_val:
                                if transition_probs[k][j] > 0 and (nl[k][i-1] + tf.math.log(transition_probs[k][j])) >= max_val:
                                    max_val = nl[k][i - 1] + tf.math.log(transition_probs[k][j])
                                    best_prev_prob = nl[k][i-1]
                                    k_new = k
                            prev_prob = best_prev_prob
                            prev_state = tf.gather(states, k_new)
                        elif i >= 1:
                            prev_prob = nl[j][i - 1]
                            prev_state = tf.gather(states, j)

                        p = (2 * math.pi * emission_paras[:, :, 1]) ** (-0.5) * \
                        tf.exp(-(evidence_vector[i] - emission_paras[:, :, 0])**2 / (2 * emission_paras[:, :, 1]**2))

                        log_p = tf.keras.ops.nan_to_num(
                            tf.math.log(p), nan=0.0, posinf=None, neginf=None
                        )                        

                        nl = tf.tensor_scatter_nd_update(nl, [[j, i]], [prev_prob + tf.reduce_sum(log_p)])
                
                new_s = tf.constant([])
                seq = tf.constant([], shape=(0, 2), dtype=tf.float32)

                highest_prob = tf.reduce_max(nl[:, -1])
                highest_prob_index = tf.argmax(nl[:, -1])
                
                new_s = tf.concat([new_s, tf.expand_dims(highest_prob, axis=0)], axis = 0)
                sequence = tf.concat([sequence, tf.expand_dims(tf.cast(highest_prob_index, dtype=tf.int32), axis=0)], axis = 0)
                sequence = tf.concat([sequence, tf.expand_dims(tf.cast(highest_prob_index, dtype=tf.int32), axis=0)], axis = 0)
                
                probability = highest_prob

                for i in range(tf.shape(evidence_vector)[0]-2,-1,-1):
                    tf.autograph.experimental.set_loop_options(
                        shape_invariants=[(sequence, tf.TensorShape([None]))]
                    )
                    change_j = None
                    highest_prob = -math.inf
                    new_highest_prob = -math.inf
                    best_state = -1
                    nj = -1
                    ni = -1

                    for j in range(len(states)):
                        if tf.reduce_any(tf.gather(transition_probs, [j, sequence[0]]) == 0):
                            sequence = tf.concat([sequence, tf.expand_dims(tf.cast(highest_prob_index, dtype=tf.int32), axis=0)], axis = 0)
                            continue
                        if tf.reduce_any((nl[j][i] + tf.math.log(tf.gather(transition_probs, [j, sequence[0]]))) > highest_prob):
                            sequence = tf.concat([sequence, tf.expand_dims(tf.cast(j, dtype=tf.int32), axis=0)], axis = 0)

                            highest_prob = nl[j][i] + tf.math.log(tf.gather(transition_probs, [j, sequence[0]]))
                            new_highest_prob = nl[j][i]
                            best_state = j
                            change_j = j
                            nj = j
                            ni = i

                    if tf.reduce_any(best_state == -1):
                        continue

                    best_state_tensor = tf.convert_to_tensor(best_state, dtype=tf.int32)
                    sequence = tf.concat([tf.expand_dims(tf.cast(best_state, dtype=tf.int32), axis=0), sequence], axis = 0)


                return sequence









from read_hmm import states_, prior_probs_tensor, transition_probs_, emission_paras_
from read_vector import vector
import tensorflow as tf
import torch

model = HMMModel(states=states_, prior_probs=prior_probs_tensor, transition_probs=transition_probs_, emission_paras=emission_paras_)

input_shape = (None, 20)
model.build(input_shape)

most_likely_sequence1 = model(evidence_vector=torch.tensor(vector[20:22]))

model.export("saved_model")

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("complete")

print("most_likely_sequence:")
print(most_likely_sequence1)
tf.print(most_likely_sequence1)


most_likely_sequence1 = model(evidence_vector=torch.tensor(vector[0:20]))
print("most_likely_sequence:")
print(most_likely_sequence1)


tf.print(most_likely_sequence1)

