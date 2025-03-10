import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import time
import gc
import argparse
from datetime import datetime
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from DeepQLearning import DeepQLearning

def create_mountaincar_model(input_shape, action_size):
    """
    Create a neural network model for the MountainCar environment
    """
    model = Sequential()
    model.add(Dense(24, activation='relu', input_dim=input_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def train_deep_q_agent(run_id=1, max_episodes=5000, max_steps=1000, render=False):
    """
    Train a Deep Q-Learning agent for the MountainCar environment
    """
    # Set random seeds for reproducibility
    np.random.seed(run_id)
    tf.random.set_seed(run_id)
    
    # Create environment
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    
    # Get state and action space info
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create model
    model = create_mountaincar_model(state_size, action_size)
    
    # Hyperparameters
    gamma = 0.99           # Discount factor
    epsilon = 1.0          # Exploration rate
    epsilon_min = 0.01     # Minimum exploration rate
    epsilon_decay = 0.995  # Decay rate for exploration
    batch_size = 64        # Size of each training batch
    memory_size = 10000    # Size of memory buffer
    
    # Create memory buffer
    memory = deque(maxlen=memory_size)
    
    # Create agent
    agent = DeepQLearning(
        env=env,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_dec=epsilon_decay,
        episodes=max_episodes,
        batch_size=batch_size,
        memory=memory,
        model=model,
        max_steps=max_steps
    )
    
    # Create directories for saving results
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Train the agent
    print(f"Starting Deep Q-Learning training run {run_id}...")
    start_time = time.time()
    
    rewards = agent.train()
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Save model
    model_filename = f"models/deepq_mountaincar_model_run{run_id}.keras"
    agent.model.save(model_filename)
    
    # Save learning curve data
    curve_filename = f"results/learning_curve_deepq_mountaincar_run{run_id}.csv"
    with open(curve_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "CumulativeReward"])
        for ep, r in enumerate(rewards, start=1):
            writer.writerow([ep, r])
    
    # Calculate success metrics
    success_episodes = [i for i, r in enumerate(rewards) if r > -200]
    success_rate = len(success_episodes) / max_episodes * 100 if max_episodes > 0 else 0
    
    print(f"Run {run_id} completed. Success rate: {success_rate:.2f}%")
    
    # Clean up to prevent memory issues
    del agent
    del model
    del memory
    env.close()
    gc.collect()
    tf.keras.backend.clear_session()
    
    return rewards

def run_multiple_experiments(runs=5, episodes=1000, steps=1000):
    """
    Run multiple Deep Q-Learning experiments and save combined results
    """
    all_rewards = []
    
    for run in range(1, runs+1):
        print(f"\n=== Starting Deep Q-Learning Run {run}/{runs} ===")
        rewards = train_deep_q_agent(run_id=run, max_episodes=episodes, max_steps=steps)
        all_rewards.append(rewards)
    
    # Convert to numpy array for easier manipulation
    all_rewards = np.array(all_rewards)
    
    # Calculate statistics
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    # Save combined data
    combined_data = pd.DataFrame({
        'Episode': range(1, episodes+1),
        'MeanReward': mean_rewards,
        'StdReward': std_rewards
    })
    combined_data.to_csv('results/deepq_mountaincar_combined.csv', index=False)
    
    # Generate and save a plot
    plot_combined_results(combined_data, 'Deep Q-Learning', 'results/deepq_mountaincar_learning_curve.png')
    
    print("\n=== Deep Q-Learning Experiment Completed ===")
    print(f"Results saved to 'results/deepq_mountaincar_combined.csv'")
    print(f"Plot saved to 'results/deepq_mountaincar_learning_curve.png'")

def plot_combined_results(data, algorithm_name, save_path, window=100):
    """
    Plot the learning curve with mean and standard deviation
    """
    # Apply moving average
    data['MeanReward_MA'] = data['MeanReward'].rolling(window=window).mean()
    data['StdReward_MA'] = data['StdReward'].rolling(window=window).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['Episode'], data['MeanReward_MA'], label=f'{algorithm_name} Mean')
    plt.fill_between(
        data['Episode'], 
        data['MeanReward_MA'] - data['StdReward_MA'], 
        data['MeanReward_MA'] + data['StdReward_MA'], 
        alpha=0.2
    )
    
    plt.axhline(y=-110, color='r', linestyle='--', label='Solved Threshold')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward (Moving Average)')
    plt.title(f'{algorithm_name} Learning Curve - MountainCar')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q-Learning for MountainCar")
    parser.add_argument('--runs', type=int, default=5, help="Number of training runs")
    parser.add_argument('--episodes', type=int, default=5000, help="Number of episodes per run")
    parser.add_argument('--steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--render', action='store_true', help="Render the environment")
    args = parser.parse_args()
    
    # Print start time
    print(f"Starting Deep Q-Learning experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.render:
        # If rendering, only do a single run
        train_deep_q_agent(run_id=1, max_episodes=args.episodes, max_steps=args.steps, render=True)
    else:
        # Otherwise, run multiple experiments
        run_multiple_experiments(runs=args.runs, episodes=args.episodes, steps=args.steps)
    
    # Print end time
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")