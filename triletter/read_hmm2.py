class HMM:
    def __init__(self, path):
        self.states = {}
        self.transition_probs = {}
        self.prior_probs = {}
        self.emission_paras = {}
        self.path = path
        self.transition_defs = {}

    def read_file_new(self):
        emission_paras_mean = {}
        emission_paras_variance = {}
        file = open(self.path, "r")
        next_line = file.readline()
        mean_defs = {}
        variance_defs = {}
        def convert_list_str_to_float(array):
            result = []
            for i in range(len(array)):
                element = float(array[i])
                result.append(element)
            return result
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
            if "~t" in next_line:
                state = next_line[len('~t "'):-2]
                next_line = file.readline()
                if "<TRANSP>" not in next_line:
                    input(f"<TRANSP> expected but not found in line: {next_line}")
                transition_length = int(next_line[len('<TRANSP> '):-1])
                self.transition_defs[state] = read_transition_probs(transition_length)
            if '~s' in next_line:
                state = next_line[len('~s "'):-2]
                next_line = file.readline()
                if "<MEAN>" not in next_line:
                    input(f"<MEAN> expected but not found in line: {next_line}")
                next_line = file.readline()
                mean = read_mean_var(next_line)
                mean_defs[state] = mean
                next_line = file.readline()
                if "<VARIANCE>" not in next_line:
                    input(f"<VARIANCE> expected but not found in line: {next_line}")
                next_line = file.readline()
                variance = read_mean_var(next_line)
                variance_defs[state] = variance
                next_line = file.readline()
                if "<GCONST>" not in next_line:
                    input(f"<GCONST> expected but not found in line: {next_line}")
            if '~h' in next_line:
                emission_paras_mean_single = {}
                emission_paras_variance_single = {}
                hmm = next_line[len('~h "'):-2]
                self.states[hmm] = []
                next_line = file.readline()
                if "<BEGINHMM>" not in next_line:
                    input(f"<BEGINHMM> expected but not found in line: {next_line}")
                next_line = file.readline()
                if "<NUMSTATES>" not in next_line:
                    input(f"<NUMSTATES> expected but not found in line: {next_line}")
                num_states = int(next_line[len("<NUMSTATES> "):-1])
                next_line = file.readline()
                for i in range(num_states - 2):
                    self.states[hmm].append(hmm + str(i))
                    if "<STATE>" not in next_line:
                        input(f"<STATE> expected but not found in line: {next_line}")
                    next_line = file.readline()
                    if "~s" in next_line:
                        state = next_line[len('~s "'):-2]
                        emission_paras_mean_single[hmm + str(i)] = mean_defs[state]
                        emission_paras_variance_single[hmm + str(i)] = variance_defs[state]
                        next_line = file.readline()
                    else:
                        if "<MEAN>" not in next_line:
                            input(f"<MEAN> expected but not found in line: {next_line}")
                        next_line = file.readline()
                        mean = read_mean_var(next_line)
                        emission_paras_mean_single[hmm + str(i)] = mean_defs[state]
                        next_line = file.readline()
                        if "<VARIANCE>" not in next_line:
                            input(f"<VARIANCE> expected but not found in line: {next_line}")
                        next_line = file.readline()
                        variance = read_mean_var(next_line)
                        emission_paras_variance_single[hmm + str(i)] = variance_defs[state]
                        next_line = file.readline()
                        if "<GCONST>" not in next_line:
                            input(f"<GCONST> expected but not found in line: {next_line}")
                        next_line = file.readline()
                if '~t "' not in next_line:
                    if "<TRANSP>" not in next_line:
                        input(f"~t or <TRANSP> expected but not found in line: {next_line}")
                    transition_length = int(next_line[len('<TRANSP> '):-1])
                    self.transition_defs[state] = read_transition_probs(transition_length)
                    self.transition_probs[hmm] = state
                else:
                    transition_state = next_line[len('~t "'):-2]
                    self.transition_probs[hmm] = transition_state
                next_line = file.readline()
                if '<ENDHMM>' not in next_line:
                    input(f"<ENDHMM> expected but not found in line: {next_line}")
                emission_paras_mean[hmm] = emission_paras_mean_single
                emission_paras_variance[hmm] = emission_paras_variance_single
            next_line = file.readline()
            for h in emission_paras_mean:
                self.emission_paras[h] = {}
                for state in self.states[h]:
                    self.emission_paras[h][state] = []
                    means = emission_paras_mean[h][state]
                    variances = emission_paras_variance[h][state]
                    for i in range(len(means)):
                        self.emission_paras[h][state].append((means[i], variances[i]))
    def change_format(self):
        new_emission_paras = {}
        for hmm in self.emission_paras:
            paras = self.emission_paras[hmm]
            new_paras = {}
            for state in paras:
                state_num = int(state[-1]) + 1
                new_paras[state_num] = paras[state]
            new_emission_paras[hmm] = new_paras
        self.emission_paras = new_emission_paras
        for i in self.states:
            for j in range(len(self.states[i])):
                self.states[i][j] = j + 1
            # self.states[i].append(len(self.states[i]) + 1)
