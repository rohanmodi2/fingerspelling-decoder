def read_file_new(path, states, emission_paras_mean, emission_paras_variance, transition_probs):
    file = open(path, "r")
    next_line = file.readline()
    count = 0
    mean_defs = {}
    variance_defs = {}
    transition_defs = {}
    all_hmms = []
    
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
            if "<TRANSP>" in next_line:
                transition_length = int(next_line[len('<TRANSP> '):-1])
                transition_defs[state] = read_transition_probs(transition_length)
                # print("transition:", state)
            else:
                pass
        if '~s' in next_line:
            state = next_line[len('~s "'):-2]
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
            for i in range(num_states - 2):
                states.append(hmm + str(i))
                next_line = file.readline()
                if "<STATE>" not in next_line:
                    input(f"<STATE> expected but not found in line: {next_line}")
                next_line = file.readline()
                if "~s" in next_line:
                    state = next_line[len('~s "'):-2]
                    print("~s FOUND MEAN DEF:", mean_defs[state])
                    print("~s FOUND VARIANCE DEF:", variance_defs[state])
                else:
                    if "<MEAN>" not in next_line:
                        input(f"<MEAN> expected but not found in line: {next_line}")
                    next_line = file.readline()
                    mean = read_mean_var(next_line)
                    print("Mean:", mean)
                    next_line = file.readline()
                    if "<VARIANCE>" not in next_line:
                        input(f"<VARIANCE> expected but not found in line: {next_line}")
                    next_line = file.readline()
                    variance = read_mean_var(next_line)
                    print("Variance:", variance)
                    next_line = file.readline()
                    if "<GCONST>" not in next_line:
                        input(f"<GCONST> expected but not found in line: {next_line}")
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
prior_probs_ = {}
path_ = "/Users/rohan/Downloads/newMacros (2)"
read_file_new(path_, states_, emission_paras_mean_, emission_paras_variance_, transition_probs_)
