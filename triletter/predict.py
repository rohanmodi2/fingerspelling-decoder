from read_hmm2 import HMM
from old_decoder import multidimensional_viterbi
from read_vector import vector
import pickle

'''hmm = HMM("/Users/rohan/Downloads/newMacros (2)")
hmm.read_file_new()
hmm.change_format()

with open('hmm.pkl', 'wb') as file:
    pickle.dump(hmm, file)'''

with open('hmm.pkl', 'rb') as file:
    hmm = pickle.load(file)  

# print("states:\n", hmm.states)
# print("emission_paras:\n", hmm.emission_paras["a"], "\n", len(hmm.emission_paras))
# print("transition_probs:\n", hmm.transition_probs)
# print(hmm.transition_defs[hmm.transition_probs['u-a+c']])

r = multidimensional_viterbi(
    evidence_vector=vector,
    states=hmm.states['h-i+m'],
    prior_probs=[None, 1, 0, 0, 0, 0, 0, 0],
    transition_probs=hmm.transition_defs[hmm.transition_probs['h-i+m']],
    emission_paras=hmm.emission_paras['h-i+m'],
    ndim=20
)

print(r)
print(len(vector))

initial_hmms = [] # a+b
secondary_hmms = [] # a-b+c

highest_prob = -1
best_hmm = None
best_sequence = None
for hmm in initial_hmms:
    sequence, prob = multidimensional_viterbi(
        evidence_vector=vector,
        states=hmm.states[hmm],
        prior_probs=[None, 1, 0, 0, 0, 0, 0, 0],
        transition_probs=hmm.transition_defs[hmm.transition_probs[hmm]],
        emission_paras=hmm.emission_paras[hmm],
        ndim=20
    )
    if prob > highest_prob:
        highest_prob = prob
        best_hmm = hmm
        best_sequence = sequence

new_vector = []
for i in range(len(best_sequence)):
    if best_sequence[i] == 7:
        new_vector = vector[i:]
        break

while new_vector:
    highest_prob = -1
    best_hmm = None
    best_sequence = None
    for hmm in secondary_hmms:
        sequence, prob = multidimensional_viterbi(
            evidence_vector=new_vector,
            states=hmm.states[hmm],
            prior_probs=[None, 1, 0, 0, 0, 0, 0, 0],
            transition_probs=hmm.transition_defs[hmm.transition_probs[hmm]],
            emission_paras=hmm.emission_paras[hmm],
            ndim=20
        )
        if prob > highest_prob:
            highest_prob = prob
            best_hmm = hmm
            best_sequence = sequence
    new_vector = []
    for i in range(len(best_sequence)):
        if best_sequence[i] == 7:
            new_vector = vector[i:]
            break