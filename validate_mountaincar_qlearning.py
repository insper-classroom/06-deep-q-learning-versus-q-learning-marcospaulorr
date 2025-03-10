import gymnasium as gym
import numpy as np
import argparse
import os
import time

def load_q_table(filename, shape):
    """
    Load a Q-table from a CSV file and reshape it
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Q-table file {filename} not found")
    
    q_table_2d = np.loadtxt(filename, delimiter=",")
    return q_table_2d.reshape(shape[0], shape[1], -1)

def discretize_state(state, env, n_states):
    """
    Discretize continuous state space into buckets
    """
    state_adj = (state - env.observation_space.low) * np.array([n_states[0], n_states[1]]) / (env.observation_space.high - env.observation_space.low)
    state_adj = np.round(state_adj, 0).astype(int)
    return np.clip(state_adj, 0, [n_states[0]-1, n_states[1]-1])

def validate_agent(q_table_file, episodes=100, max_steps=1000, render=False):
    """
    Validate a trained Q-Learning agent
    """
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    
    # Discretize the state space (same as in training)
    n_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
    n_states = np.round(n_states, 0).astype(int) + 1
    
    # Load the Q-table
    q_table = load_q_table(q_table_file, n_states)
    
    # Statistics variables
    rewards_list = []
    steps_list = []
    success_count = 0
    position_reached = []
    
    print(f"Starting validation with {episodes} episodes...")
    
    for episode in range(1, episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Discretize state
            state_disc = discretize_state(state, env, n_states)
            
            # Select action (greedy policy)
            action = np.argmax(q_table[state_disc[0], state_disc[1]])
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Break if done
            if done:
                break
        
        # Check if solved (reached the flag)
        if state[0] >= 0.5:
            success_count += 1
        
        # Record stats
        rewards_list.append(episode_reward)
        steps_list.append(steps)
        position_reached.append(state[0])
        
        # Print progress
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode}/{episodes} - Reward: {episode_reward:.2f}, Steps: {steps}, "
                  f"Position: {state[0]:.4f}")
    
    # Calculate statistics
    success_rate = (success_count / episodes) * 100
    avg_reward = np.mean(rewards_list)
    avg_steps = np.mean(steps_list)
    avg_position = np.mean(position_reached)
    
    print("\nValidation Results:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Position Reached: {avg_position:.4f}")
    
    env.close()
    return success_rate, avg_reward, avg_steps, avg_position

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Q-Learning agent for MountainCar")
    parser.add_argument('--qtable', type=str, required=False, 
                        default="models/qtable_qlearning_mountaincar_run1.csv",
                        help="Path to Q-table file")
    parser.add_argument('--episodes', type=int, default=100, help="Number of episodes for validation")
    parser.add_argument('--steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--render', action='store_true', help="Render the environment")
    args = parser.parse_args()
    
    validate_agent(args.qtable, args.episodes, args.steps, args.render)