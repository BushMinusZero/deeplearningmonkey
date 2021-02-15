"""
CS234: Reinforcement Learning

Markov Decision Process

Word:


Dynamics:
    P(s_i | s_i, a1) = 0.5
    P(s_i+1 | s_i, a1) = 0.5

Rewards:
    +1 in state s1
    +10 in state s7
    0 otherwise

Actions:
    a1 - try going to the right
    a2 - don't move (deterministic)
    a3 - try going to the left
"""
import numpy as np
import random


class Example1:
    """ Practice Markov Decision Process Iteration of Policy Evaluation

    - Assume a really simple policy function pi(a)=a1
    """

    def __init__(self):

        # Initialize the discount factor
        self.states = 7
        self.gamma = 0.5

        # Initialize value
        self.V_1 = [1, 0, 0, 0, 0, 0, 10]

        # Store all values of V
        self.V = [self.V_1, ]

    def reward(self, s):
        return self.V_1[s]

    @staticmethod
    def transition_probabilities(s: int):
        """Probability transition matrix."""
        P = [0, 0, 0, 0, 0, 0, 0]
        if s == 6:
            P[6] = 1.0
        else:
            P[s] = 0.5
            P[s + 1] = 0.5
        return P

    def iterate(self):
        V_k = []
        for s in range(self.states):
            expectation = zip(self.V[-1], self.transition_probabilities(s))
            V_k_s = self.reward(s) + self.gamma * sum([v * p for v, p in expectation])
            V_k.append(V_k_s)
        return V_k

    def estimate_value_function(self, k):
        for i in range(k):
            V_i = self.iterate()
            self.V.append(V_i)
            print(V_i)
        return self.V[-1]


class Example2:
    """
    MDP (Markov Decision Process) Policy improvement

    - introduces state-action value of a policy Q
    """

    def __init__(self):
        # Initialize the discount factor
        self.states = 7
        self.gamma = 0.5
        self.value_estimation_iterations = 10

        # Initialize value
        self.V_1 = [1, 0, 0, 0, 0, 0, 10]

        # Store all values of V
        self.V = [self.V_1, ]

        # Store all policies
        self.pi = []

        # Store all possible actions
        self.actions = ["tryLeft", "stay", "tryRight"]

        # Store all possible actions for each state
        self.possible_actions = []
        for s in range(self.states):
            if s == 0:
                self.possible_actions.append(["stay", "tryRight"])
            elif s == 6:
                self.possible_actions.append(["tryLeft", "stay"])
            else:
                self.possible_actions.append(self.actions)

    def random_policy(self):
        """Create a random policy for each state
        Policies map states to actions
        """
        return [random.choice(a) for a in self.possible_actions]

    def transition_probabilities(self, s, pi):
        """Combines the policy with the stochastic nature of actions.
        e.g. rover decides to go right but only succeeds with a 50% likelihood
        :param s: current state
        :param pi: policy pi which maps state s to an action
        """
        P = [0, 0, 0, 0, 0, 0, 0]
        action = pi[s]
        if action == "tryRight":
            P[s] = 0.5
            P[s + 1] = 0.5
        elif action == "tryLeft":
            P[s] = 0.5
            P[s - 1] = 0.5
        elif action == "stay":
            P[s] = 1.0
        else:
            raise Exception("Not a valid action")
        return P

    def iterate_value_function(self, pi):
        V_k = []
        for s in range(self.states):
            expectation = zip(self.V[-1], self.transition_probabilities(s, pi))
            V_k_s = self.V_1[s] + self.gamma * sum([v * p for v, p in expectation])
            V_k.append(V_k_s)
        return V_k

    def estimate_value_function(self, k, pi):
        for i in range(k):
            V_i = self.iterate_value_function(pi)
            self.V.append(V_i)
            print(V_i)
        return self.V[-1]

    @staticmethod
    def action_effect(a, s):
        if a == "tryLeft":
            return s - 1
        elif a == "tryRight":
            return s + 1
        else:
            return s

    def iterate_policy_improvement(self, pi):

        # Compute the state-action value of policy pi_i
        Q = []
        for s in range(self.states):
            Q_row = []
            for a in self.actions:
                if a not in self.possible_actions[s]:
                    value = 0
                else:
                    s_prime = self.action_effect(a, s)
                    reward = self.V[-1][s_prime]
                    expectation = zip(self.V[-1], self.transition_probabilities(s_prime, pi))
                    value = reward + self.gamma * sum([v * p for v, p in expectation])
                Q_row.append(value)
            Q.append(Q_row)

        # Compute the new policy pi_i+1
        best_actions = np.argmax(Q, axis=1).tolist()
        new_pi = [self.actions[i] for i in best_actions]
        self.pi.append(new_pi)

    def policy_iteration(self):

        num_iterators = 10

        # Initialize policy randomly for all states s
        i = 0
        pi_0 = self.random_policy()
        self.pi.append(pi_0)

        # Compute the value of that policy
        self.estimate_value_function(self.value_estimation_iterations, pi_0)

        while i <= num_iterators:
            self.iterate_policy_improvement(self.pi[-1])
            self.estimate_value_function(self.value_estimation_iterations, self.pi[-1])
            i += 1
            print(self.pi[-1])


def main():

    e = Example2()
    e.policy_iteration()


main()
