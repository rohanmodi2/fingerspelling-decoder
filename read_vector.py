import numpy as np


def read_vector(filepath):
    vector = []
    count = 0
    file = open(filepath, "r")
    line = file.readline()
    while line:
        line = np.array(line.split("  "))
        line = line.astype(float)
        # print(line)
        import sys
        vector.append(tuple(line))
        line = file.readline()
        count += 1
    return vector


vector = read_vector('/Users/rohan/Downloads/1963838355')
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    from decoder import multidimensional_viterbi
    from read_hmm import states_, prior_probs_, transition_probs_, emission_paras_

    vector = read_vector('/Users/rohan/Downloads/1960228537')

    a = multidimensional_viterbi(vector, states_, prior_probs_, transition_probs_, emission_paras_, ndim=20)[0]
    last_letter = None
    result_str = ""
    for i in range(len(a)):
        if a[i][:-1] == last_letter:
            continue
        print(f"{i}: {a[i][:-1]}")
        last_letter = a[i][:-1]
        
        result_str += a[i][:-1] if a[i][:-1] not in ["sil0", "sil1"] else a[i][-2]
    print("result str:")
    print(result_str)



"""
(difflib.SequenceMatcher(None, "0zxsutoksixmons1", "0xsvxoksixmonks").ratio())

1963838355:
- htk: 0zxsutoksixmons1
- new model: 0xsvxoksixmonks
- actual: 0the_assault_took_six_months1
-- score (htk): 0.7741935483870968
-- score (actual): 0.5
-- htk / actual score: 0.6222222222222222

196531928:
- htk: 0esixsrxhtdsndeusos1
- new model: emixoarxhnmanomvmsonsb
- actual: 0six_daughters_and_seven_sons1
-- score (htk): 0.47619047619047616
-- score (actual): 0.4230769230769231
-- htk / actual score: 0.56

1960228537:
- htk: 0hdsyd1
- new model: 
- actual: 0her_majesty_visited_our_country1
-- score (htk): 0.5454545454545454
-- score (actual): 0.21621621621621623
-- htk / actual score: 0.3

"""

