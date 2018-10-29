""" Implements the Viterbi algorithm """

import numpy as np


def viterbi(hmm, string_to_decode):
    """
    Decode string using HMM.
    :param hmm: HiddenMarkovModel object describing HMM to use
    :param string_to_decode: String to be decoded
    :return: (state_sequence, probability) Tuple of decoded state
             sequence and its probability
    """
    N = len(hmm.states)
    T = len(string_to_decode)

    viterbi = np.zeros((N+2, T)) # The probability matrix
    backpointer = np.zeros((N+2, T)) # The backpointers matrix

    # Initialization
    for i in range(N):
        viterbi[i][0] = hmm.initial_probability[hmm.states[i]] * hmm.emission_probability[hmm.states[i]][string_to_decode[0]]
        backpointer[i][0] = '-1'

    for t in range(1, T):
        for s in range(N):
            _probabilities = [viterbi[i][t-1] * hmm.transition_probability[hmm.states[i]][hmm.states[s]] for i in range(N)]
            best_prev = max(enumerate(_probabilities), key=lambda x:x[1])
            viterbi[s][t] = best_prev[1] * hmm.emission_probability[hmm.states[s]][string_to_decode[t]]
            backpointer[s][t] = best_prev[0]

    # Final winner
    _probabilities = [viterbi[i][T-1] * 1 for i in range(N)]
    best_prev = max(enumerate(_probabilities), key=lambda x:x[1])
    back_trace = [hmm.states[best_prev[0]]]
    _curr = best_prev[0]

    for t in range(T-1,0,-1):
        _temp = int(backpointer[_curr][t])
        _curr = _temp
        back_trace.append(hmm.states[_temp])

    return back_trace[::-1], best_prev[1]
