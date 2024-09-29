def read_file_new(path, states, emission_paras_mean, emission_paras_variance, transition_probs):
    file = open(path, "r")
    next_line = file.readline()
    count = 0
    while next_line:
        if "~h " in next_line:
            letter = next_line[len('~h "')]  # only works for single letters
            count += 1
            if "sil0" in next_line:
                letter = "sil0"
            elif "sil1" in next_line:
                letter = "sil1"
            # print(f"letter: {letter}")
            next_line = file.readline()
            if "<BEGINHMM>" not in next_line:
                input(f"<BEGINHMM> expected but not found in line: {next_line}")
            next_line = file.readline()
            num_states = next_line[len("<NUMSTATES> ")]
            num_states = int(num_states)
            # print(f"num_states: {num_states}")
            for i in range(2, num_states):
                next_line = file.readline()
                if "<STATE>" not in next_line:
                    input(f"<STATE> expected but not found in line: {next_line}")
                state_number = int(next_line[len("<STATE> "):])
                # print(f"state_number: {state_number}")
                states.append(f"{letter}{state_number}")
                next_line = file.readline()
                if "<MEAN>" not in next_line:
                    input(f"<MEAN> expected but not found in line: {next_line}")
                next_line = file.readline()
                next_line = next_line.strip()
                mean_list = next_line.split(" ")
                mean_list = convert_list_str_to_float(mean_list)
                # print(f"mean_list: {mean_list}")
                emission_paras_mean[f"{letter}{state_number}"] = mean_list

                next_line = file.readline()
                if "<VARIANCE>" not in next_line:
                    input(f"<VARIANCE> expected but not found in line: {next_line}")
                next_line = file.readline()
                next_line = next_line.strip()
                variance_list = next_line.split(" ")
                variance_list = convert_list_str_to_float(variance_list)
                # print(f"variance_list: {variance_list}")
                emission_paras_variance[f"{letter}{state_number}"] = variance_list

                next_line = file.readline()
                if "<GCONST>" not in next_line:
                    input(f"<GCONST> expected but not found in line: {next_line}")
            states.append(f"{letter}{num_states}")
            emission_paras_mean[f"{letter}{num_states}"] = [None] * 20
            emission_paras_variance[f"{letter}{num_states}"] = [None] * 20
            next_line = file.readline()
            if "<TRANSP>" not in next_line:
                input(f"<TRANSP> expected but not found in line: {next_line}")
            len_transp = int(next_line[len("<TRANSP> "):])
            transp_list = []
            for j in range(len_transp):
                next_line = file.readline()
                if j == 0:  # skipping state 1 (start state)
                    continue
                transition_probs[f"{letter}{j+1}"] = {}
                transp_line = next_line.strip()
                transp_line = transp_line.split(" ")
                transp_line = convert_list_str_to_float(transp_line)
                for k in range(len(transp_line)):
                    transition_probs[f"{letter}{j+1}"][f"{letter}{k+1}"] = transp_line[k]
                transp_list.append(transp_line)
            # print(f"transp_list:\n{transp_list}")
            # transition_probs[f"{letter}"] = transp_list
            next_line = file.readline()
            if "<ENDHMM>" not in next_line:
                input(f"<ENDHMM> expected but not found in line: {next_line}")
        next_line = file.readline()
    # print("COUNT: ", count)


def convert_list_str_to_float(array):
    result = []
    for i in range(len(array)):
        element = float(array[i])
        result.append(element)
    return result


states_ = []
emission_paras_ = {}
emission_paras_mean_ = {}
emission_paras_variance_ = {}
transition_probs_ = {}
prior_probs_ = {}
path_ = "/Users/rohan/Downloads/6state_uniletter_1mix.tar.gz/models/hmm0.19/newMacros"
read_file_new(path_, states_, emission_paras_mean_, emission_paras_variance_, transition_probs_)

for state_ in states_:
    if state_[-1] == '2':
        prior_probs_[state_] = 1/29
    else:
        prior_probs_[state_] = 0


for t in transition_probs_:
    if t[-1] == '7' or t in ['sil04', 'sil14']:
        transition_probs_[t] = prior_probs_


for state in states_:
    emission_paras_[state] = []
    means = emission_paras_mean_[state]
    variances = emission_paras_variance_[state]
    for i in range(len(means)):
        emission_paras_[state].append((means[i], variances[i]))

prior_probs_list = []
for state in states_:
    prior_probs_list.append(prior_probs_[state])


import torch
prior_probs_tensor = torch.Tensor(prior_probs_list)


