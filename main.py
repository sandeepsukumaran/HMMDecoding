""" Decode a given input string according to hard-coded HMM """

import argparse
import numpy as np
from HMM import HiddenMarkovModel
from Viterbi import viterbi

if __name__ == "__main__":

    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", help="Increase output verbosity", action="store_true")
    parser.add_argument("string_to_decode", help="The string to be decoded")
    args = parser.parse_args()
    input_string = args.string_to_decode

    # Define the HMM
    h = HiddenMarkovModel(['HOT', 'COLD'], ['1', '2', '3'])

    # Initial probabilities
    initial_probabilities = [0.8, 0.2]

    # Transition probabilities
    transition_probabilities = [[0.7, 0.3], [0.6, 0.4]]

    # Emission probabilities
    emission_probabilities = [[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]]

    # Set up probabilites
    h.set_probabilities(initial_probabilities, transition_probabilities, emission_probabilities)

    # Run Viterbi algorithm
    trace, prob = viterbi(h, input_string)

    print("Trace:")
    print(trace)

    print("Probability:")
    print(prob)
