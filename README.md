# VALUE ITERATION ALGORITHM

## AIM

To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT

The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment

Environment Setup:

The code begins by importing necessary libraries and setting up the Frozen Lake environment using Gym. It also initializes the initial state, goal state, and transition probabilities (P).

### Value Iteration Algorithm:

1.The core algorithm used in this code is Value Iteration. Value Initialization:

2.Initialize a value function (V) with zeros for each state. Value Iteration Loop:

3.For each state (s), calculate the Q-values for all possible actions (a) using the Bellman equation. The Q-value represents the expected cumulative reward when taking action 'a' from state 's'.

4.Update the value function (V) by taking the maximum Q-value for each state.

5.Check for convergence: If the maximum change in value function (V) is smaller than a threshold (theta), break the loop.

6.After convergence, derive the optimal policy (pi) by selecting actions that maximize the Q-values.

Results:

The code prints the optimal policy, its state-value function, the success rate of reaching the goal, and the mean undiscounted return of the optimal policy.

## VALUE ITERATION FUNCTION
~~~
Developed by:Lakshman
Reg no:212222240001
~~~

~~~python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        # Initialize the action-value function Q as an array of zeros
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    # Update the action-value function Q using the Bellman equation
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        # Check if the maximum difference between Old V and new V is less than theta.
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break

        # Update the value function V with the maximum action-value from Q
        V = np.max(Q, axis=1)

    # Compute the policy pi based on the action-value function Q
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi
~~~

## OUTPUT:

Mention the optimal policy, optimal value function , success rate for the optimal policy.

### Optimal policy
![Screenshot 2024-04-26 172915](https://github.com/LakshmanAdhireddy/rl-value-iteration/assets/118707265/ce904b70-ee8f-41c3-b672-122893586b85)


### Optimal value function
![Screenshot 2024-04-26 173014](https://github.com/LakshmanAdhireddy/rl-value-iteration/assets/118707265/2a2ebe11-77f7-46b5-ba03-9245223f3481)

## success rate
![Screenshot 2024-04-26 172929](https://github.com/LakshmanAdhireddy/rl-value-iteration/assets/118707265/c96cab08-253f-4852-a957-0e2ea89c44c6)

## RESULT:

Therefore a python program has been successfully developed to find the optimal policy for the given MDP using the value iteration algorithm.
