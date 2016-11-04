#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state

all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE

    np.set_printoptions(precision=3, linewidth=200) 

    trans_matrix = np.array([[0.0] * 440] * 440)
    obs_matrix = np.array([[0.0] * 96] * 440)    
    
    num_states = len(all_possible_hidden_states)
    
    #populate trans_matrix
    for i in range(num_states):
        hidden_state = all_possible_hidden_states[i]
        transitions = transition_model(hidden_state)
        trans_matrix[i] = np.array([transitions[k] if k in 
            transitions else 0.0 for k in all_possible_hidden_states])
  
    #populate obs_matrix
    for i in range(num_states):
        hidden_state = all_possible_hidden_states[i]             
        obs = observation_model(hidden_state)
        obs_matrix[i] = np.array([obs[k] if k in 
            obs else 0.0 for k in all_possible_observed_states])
       
    """
    print(trans_matrix[:10, :10], '\n')
    print(obs_matrix[:10, :10], '\n')
    
    #print(observations)
    print("\n###########\n")
    #print(all_possible_hidden_states[:10])
    #print("\n###########\n")
    print(len(prior_distribution))
    #print("\n###########\n")
    #print(prior_distribution)
    print("\n###########\n")
    print(transition_model(all_possible_hidden_states[13]))
    print("\n###########\n")
    print(all_possible_hidden_states[13])
    print(observation_model(all_possible_hidden_states[13]))
    print("\n###########\n")
    """
    #

    
    
    # TODO: Compute the forward messages
    #print(prior_distribution)
    #print(len(all_possible_hidden_states))
    
    A = trans_matrix
    B = obs_matrix
    prior = np.array([prior_distribution[k] if k in prior_distribution else 0.0 
                    for k in all_possible_hidden_states])   
    states = all_possible_observed_states
    """
    
    A = np.array([[0.25, 0.75, 0],[0, 0.25, 0.75],[0,0,1.0]])                    
    B = np.array([[1.0,0],[0,1.0],[1.0,0]])
    prior = np.array([1.0/3] * 3) 
    states = ["hot", "cold"] 
    observations = ["hot", "cold", "hot"]   
    """
    num_time_steps = len(observations)
    forward_messages = [None] * (num_time_steps + 1)
    forward_messages[0] = prior
    
    for i in range(0, num_time_steps):
      prev_message = forward_messages[i]
        
      obs = observations[i]
      
      if obs != None:
          obs_index = states.index(obs)
          xk = B[:,obs_index]   
      else:
          xk = np.ones(len(all_possible_hidden_states))
      
      step1 = prev_message * xk                 
      
      unnormal = np.matmul(step1, A) 
      forward_messages[i + 1] = unnormal / np.sum(unnormal)

      
    #print(forward_messages)
    #debug
    """
    msgno = 2
    msg = forward_messages[msgno]
    norm = msg/msg.sum()
    print('Normalizing factor forward = {} for message {}'.format(msg.sum(), msgno))
    print([(i, m) for i, m in enumerate(norm) if m != 0][:4])
    print([(all_possible_hidden_states[i], m) for i, m in enumerate(norm) if m != 0][:4], '\n\n')
    """

    backward_messages = [None] * (num_time_steps + 1)
    terminal = np.ones(len(all_possible_hidden_states))
    
    backward_messages[num_time_steps] = terminal
    # TODO: Compute the backward messages
    for i in range(num_time_steps - 1, -1, -1):
      
      prev_message = backward_messages[i + 1]
        
      obs = observations[i]
      
      if obs != None:
          obs_index = states.index(obs)
          xk = B[:,obs_index]    
      else:
          xk = np.ones(len(all_possible_hidden_states))
      
      step1 = prev_message * xk                 
      
      unnormal = np.matmul(step1, A.transpose()) 
      backward_messages[i] = unnormal / np.sum(unnormal)    
    
    """
    msgno = 98
    msg = backward_messages[msgno]
    norm = msg/msg.sum()
    print('Normalizing factor forward = {} for message {}'.format(msg.sum(), msgno))
    print([(i, m) for i, m in enumerate(norm) if m != 0][:4])
    print([(all_possible_hidden_states[i], m) for i, m in enumerate(norm) if m != 0][:4], '\n\n')
    """
    #print("message length sanity check")
    #print(len(forward_messages))
    #print(len(backward_messages))
    
    #print(backward_messages[:10])    
    
    marginals = [None] * num_time_steps
    #marginals = np.array([[0] * len(all_possible_hidden_states)] * num_time_steps)
    for i, obs in enumerate(observations):
      
      if obs != None:
          x_i = states.index(obs)
          em = B[:, x_i]
      else:
          em = np.ones(len(all_possible_hidden_states))
      
      
      """"print(x_i)
      print(forward_messages[i])
      print(backward_messages[i])
      print(B[:, x_i])"""
      marginal = forward_messages[i] * backward_messages[i + 1] * em  
      
      dist = robot.Distribution()
      for j, hstate in enumerate(all_possible_hidden_states):
        if marginal[j] != 0:
          dist[hstate] = marginal[j]
      dist.renormalize()
      marginals[i] = dist


    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    np.set_printoptions(precision=3, linewidth=200) 

    A = np.array([[0.0] * 440] * 440)
    B = np.array([[0.0] * 96] * 440)    
    
    num_states = len(all_possible_hidden_states)
    
    #populate trans_matrix
    for i in range(num_states):
        hidden_state = all_possible_hidden_states[i]
        transitions = transition_model(hidden_state)
        A[i] = np.array([transitions[k] if k in 
            transitions else 0.0 for k in all_possible_hidden_states])
  
    #populate obs_matrix
    for i in range(num_states):
        hidden_state = all_possible_hidden_states[i]             
        obs = observation_model(hidden_state)
        B[i] = np.array([obs[k] if k in 
            obs else 0.0 for k in all_possible_observed_states])

    #calculate prior distribution
    prior = np.array([prior_distribution[k] if k in prior_distribution else 0.0 
                    for k in all_possible_hidden_states])  

    estimated_hidden_states = run_viterbi(A, B, prior, 
                                          all_possible_hidden_states,
                                          all_possible_observed_states,
                                          observations)

    return estimated_hidden_states

def run_viterbi(A, B, prior, all_hstates, all_obs, observations):
    log_prior = np.log2(prior)
    log_a = np.log2(A)
    log_b = np.log2(B)
    
    messages = np.array([[None] * len(all_hstates)] * (len(observations) + 1))
    messages[0] = log_prior
    back_pointers = np.array([[None] * len(all_hstates)] * (len(observations) + 1))
    
    for i, obs in enumerate(observations):
        if obs != None:
            obs_i = all_obs.index(obs)  
            em = log_b[:,obs_i]
        else:
            em = np.log2(np.ones(len(all_hstates)))            
            
        for j, state in enumerate(all_hstates):
           blarg = messages[i] + log_a.transpose()[j] + em              
           messages[i+1][j] = np.max(blarg) 
           back_pointers[i+1][j] = np.argmax(blarg)
    
    response = [None] * len(observations)
    most_likely = np.argmax(messages[-1])
    for i in range(len(back_pointers)-1, 0, -1):
        index = back_pointers[i][most_likely]
        most_likely = index
        response[i-1] = all_hstates[index]
        
    return response    


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    np.set_printoptions(precision=3, linewidth=200) 

    A = np.array([[0.0] * 440] * 440)
    B = np.array([[0.0] * 96] * 440)    
    
    num_states = len(all_possible_hidden_states)
    
    #populate trans_matrix
    for i in range(num_states):
        hidden_state = all_possible_hidden_states[i]
        transitions = transition_model(hidden_state)
        A[i] = np.array([transitions[k] if k in 
            transitions else 0.0 for k in all_possible_hidden_states])
  
    #populate obs_matrix
    for i in range(num_states):
        hidden_state = all_possible_hidden_states[i]             
        obs = observation_model(hidden_state)
        B[i] = np.array([obs[k] if k in 
            obs else 0.0 for k in all_possible_observed_states])

    #calculate prior distribution
    prior = np.array([prior_distribution[k] if k in prior_distribution else 0.0 
                    for k in all_possible_hidden_states])  

    estimated_hidden_states = run_viterbi2(A, B, prior, 
                                          all_possible_hidden_states,
                                          all_possible_observed_states,
                                          observations)

    return estimated_hidden_states

def run_viterbi2(A, B, prior, all_hstates, all_obs, observations):
    log_prior = np.log2(prior)
    log_a = np.log2(A)
    log_b = np.log2(B)
    
    messages = np.array([[[None] * len(all_hstates)] * (len(observations) + 1)]*2)
    messages[0][0] = log_prior
    messages[1][0] = log_prior
    back_pointers = np.array([[[None] * len(all_hstates)] * (len(observations) + 1)]*2)
    
    best_second = 9999999    
    
    for i, obs in enumerate(observations):
        if obs != None:
            obs_i = all_obs.index(obs)  
            em = log_b[:,obs_i]
        else:
            em = np.log2(np.ones(len(all_hstates)))            
            
        for j, state in enumerate(all_hstates):
           first = messages[0][i] + log_a.transpose()[j] + em
           second = messages[1][i] + log_a.transpose()[j] + em
           
           indicies = np.argsort(first)
           
           first_val = first[indicies[-1]]
           second_val = second[indicies[-2]]
           
           diff = first_val - second_val
           
           if diff > 0 and diff < best_second:
               #print("Previous best: " + str(best_second))
               #print("New best: " + str(diff))
               
               best_second = diff
               messages[1][:] = messages[0][:]
               back_pointers[1][:] = back_pointers[0][:]
           
               messages[0][i+1][j] = first_val
               messages[1][i+1][j] = second_val
           
               back_pointers[0][i+1][j] = indicies[-1]
               back_pointers[1][i+1][j] = indicies[-2]
           else:
               messages[0][i+1][j] = first_val
               messages[1][i+1][j] = np.max(second)
           
               back_pointers[0][i+1][j] = indicies[-1]
               back_pointers[1][i+1][j] = np.argmax(second)
        
    
    response = [None] * len(observations)
    most_likely = np.argmax(messages[1][-1])
    for i in range(len(back_pointers[1])-1, 0, -1):
        index = back_pointers[1][i][most_likely]
        most_likely = index
        response[i-1] = all_hstates[index]
        
    return response

# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
