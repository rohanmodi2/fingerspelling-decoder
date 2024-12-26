def read_file_new(path, states, emission_paras_mean, emission_paras_variance, transition_probs, transition_defs):
    file = open(path, "r")
    next_line = file.readline()
    all_hmms = []
    mean_defs = {}
    variance_defs = {}

    def read_transition_probs(n):
        result = []
        for i in range(n):
            next_line = file.readline()
            result.append(read_mean_var(next_line))
        return result
    def read_mean_var(line):
        line = line.strip()
        line = line.split(" ")
        vals = convert_list_str_to_float(line)
        return vals
    while next_line:
        # print(next_line)
        if "~t" in next_line:
            state = next_line[len('~t "'):-2]
            next_line = file.readline()
            if "<TRANSP>" not in next_line:
                input(f"<TRANSP> expected but not found in line: {next_line}")
            transition_length = int(next_line[len('<TRANSP> '):-1])
            transition_defs[state] = read_transition_probs(transition_length)
        if '~s' in next_line:
            state = next_line[len('~s "'):-2]
            print("done:", state)
            # print("state:", state)
            next_line = file.readline()
            if "<MEAN>" not in next_line:
                input(f"<MEAN> expected but not found in line: {next_line}")
            next_line = file.readline()
            mean = read_mean_var(next_line)
            # print("Mean:", mean)
            mean_defs[state] = mean
            next_line = file.readline()
            if "<VARIANCE>" not in next_line:
                input(f"<VARIANCE> expected but not found in line: {next_line}")
            next_line = file.readline()
            variance = read_mean_var(next_line)
            # print("Variance:", variance)
            variance_defs[state] = variance
            next_line = file.readline()
            if "<GCONST>" not in next_line:
                input(f"<GCONST> expected but not found in line: {next_line}")
        if '~h' in next_line:
            hmm = next_line[len('~h "'):-2]
            all_hmms.append(hmm)
            next_line = file.readline()
            if "<BEGINHMM>" not in next_line:
                input(f"<BEGINHMM> expected but not found in line: {next_line}")
            next_line = file.readline()
            if "<NUMSTATES>" not in next_line:
                input(f"<NUMSTATES> expected but not found in line: {next_line}")
            num_states = int(next_line[len("<NUMSTATES> "):-1])
            next_line = file.readline()
            for i in range(num_states - 2):
                states.append(hmm + str(i))
                if "<STATE>" not in next_line:
                    input(f"<STATE> expected but not found in line: {next_line}")
                print(next_line)
                next_line = file.readline()
                if "~s" in next_line:
                    state = next_line[len('~s "'):-2]
                    print("~s FOUND MEAN DEF:", mean_defs[state])
                    emission_paras_mean[hmm + str(i)] = mean_defs[state]
                    print("~s FOUND VARIANCE DEF:", variance_defs[state])
                    emission_paras_variance[hmm + str(i)] = variance_defs[state]
                    next_line = file.readline()
                else:
                    if "<MEAN>" not in next_line:
                        input(f"<MEAN> expected but not found in line: {next_line}")
                    next_line = file.readline()
                    mean = read_mean_var(next_line)
                    print("Mean:", mean)
                    emission_paras_mean[hmm + str(i)] = mean_defs[state]
                    next_line = file.readline()
                    if "<VARIANCE>" not in next_line:
                        input(f"<VARIANCE> expected but not found in line: {next_line}")
                    next_line = file.readline()
                    variance = read_mean_var(next_line)
                    print("Variance:", variance)
                    emission_paras_variance[hmm + str(i)] = variance_defs[state]
                    next_line = file.readline()
                    if "<GCONST>" not in next_line:
                        input(f"<GCONST> expected but not found in line: {next_line}")
                    next_line = file.readline()
            if '~t "' not in next_line:
                if "<TRANSP>" not in next_line:
                    input(f"~t or <TRANSP> expected but not found in line: {next_line}")
                transition_length = int(next_line[len('<TRANSP> '):-1])
                transition_defs[state] = read_transition_probs(transition_length)
                transition_probs[hmm] = state
            else:
                transition_state = next_line[len('~t "'):-2]
                transition_probs[hmm] = transition_state
            next_line = file.readline()
            if '<ENDHMM>' not in next_line:
                input(f"<ENDHMM> expected but not found in line: {next_line}")
        next_line = file.readline()
    # print("ALL HMMS:")
    # print(all_hmms)
    print("TRANSITION PROBS:")
    print(transition_defs)
    print(states)
    print(len(states))
    for hmm in all_hmms:
        print(hmm)
    

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
transition_defs_ = {}
prior_probs_ = {}
path_ = "/Users/rohan/Downloads/newMacros (2)"
read_file_new(path_, states_, emission_paras_mean_, emission_paras_variance_, transition_probs_, transition_defs_)

print("emission_paras_mean_:", emission_paras_mean_)

for state in states_:
    emission_paras_[state] = []
    means = emission_paras_mean_[state]
    variances = emission_paras_variance_[state]
    for i in range(len(means)):
        emission_paras_[state].append((means[i], variances[i]))

count = 0
for state_ in states_:
    if len(state_) == 2 or len(state_) == 4:
        count += 1

for state_ in states_:
    if len(state_) == 2 or len(state_) == 4:
        prior_probs_[state_] = 1/count
    else:
        prior_probs_[state_] = 0
