""" Class definition for Hidden Markov Model """


class HiddenMarkovModel:
    def __init__(self, latent_states, observations):
        """
	Set up the HMM object.
	:param latent_states: List of states
	:param observations: List of observations
	:return: None
	"""
        self.states = latent_states
        self.observations = observations
        self.initial_probability = {i: 0.0 for i in self.states} 
        self.transition_probability = {i: {j:0.0 for j in self.states} for i in self.states}
        self.emission_probability = {i: {j:0.0 for j in self.observations} for i in self.states}

    def set_probabilities(self, initial_probabilities, transition_probabilities, emission_probabilities):
        """
        Set the HMM's transition and emission probabilites.
	:param initial_probabilites: List where i th entry is probabilitt of being in ith state initially
        :param transition_probabilities: List of lists where i,j th entry contains transition probability from state i to state j
        :param emission_probabilities: List of lists where i,j the entry indicates probability of emitting jth observation in state i
        :return: None
        """
        for i in range(len(self.states)):
            self.initial_probability[self.states[i]] = initial_probabilities[i]

        for i in range(len(self.states)):
            for j in range(len(self.states)):
                self.transition_probability[self.states[i]][self.states[j]] = transition_probabilities[i][j]

        for i in range(len(self.states)):
            for j in range(len(self.observations)):
                self.emission_probability[self.states[i]][self.observations[j]] = emission_probabilities[i][j]

