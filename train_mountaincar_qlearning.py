import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import argparse
import time
from datetime import datetime

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, 
                 epsilon=0.7, epsilon_min=0.01, epsilon_decay=0.999):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states[0], n_states[1], n_actions))
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q_table[state[0], state[1]])
    
    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state[0], next_state[1]])
        td_target = reward if done else reward + self.gamma * best_next
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.alpha * td_error
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_q_table(self, filename="qtable_qlearning_mountaincar.csv"):
        rows, cols, actions = self.q_table.shape
        q_table_2d = self.q_table.reshape(rows * cols, actions)
        np.savetxt(filename, q_table_2d, delimiter=",")
    
    def load_q_table(self, filename="qtable_qlearning_mountaincar.csv", shape=None):
        if os.path.exists(filename):
            q_table_2d = np.loadtxt(filename, delimiter=",")
            if shape:
                self.q_table = q_table_2d.reshape(shape[0], shape[1], -1)
            else:
                self.q_table = q_table_2d.reshape(self.n_states[0], self.n_states[1], -1)
        else:
            print("Problema na Q-Table")

def discretize_state(state, env, n_states):
    state_adj = (state - env.observation_space.low) * np.array([n_states[0], n_states[1]]) / (env.observation_space.high - env.observation_space.low)
    state_adj = np.round(state_adj, 0).astype(int)
    return np.clip(state_adj, 0, [n_states[0]-1, n_states[1]-1])

def train_agent(run_id=1, max_episodes=5000, max_steps=1000, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    
    # Discretizando espaço
    n_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
    n_states = np.round(n_states, 0).astype(int) + 1
    n_actions = env.action_space.n
    
    agent = QLearningAgent(
        n_states=n_states, 
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.7,
        epsilon_min=0.01,
        epsilon_decay=0.999
    )
    
    # Salvando resultados
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('models'):
        os.makedirs('models')
        
    rewards_per_episode = []
    solved_episodes = []
    timestep_history = []
    

    start_time = time.time()
    
    print(f"Começando treinamento {run_id}...")
    
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        state_disc = discretize_state(state, env, n_states)
        
        total_reward = 0
        steps = 0
        done = False
        
        for step in range(max_steps):
            action = agent.select_action(state_disc)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_disc = discretize_state(next_state, env, n_states)
            
            # Atualizando Q-Table
            agent.update(state_disc, action, reward, next_state_disc, done)
            
            state = next_state
            state_disc = next_state_disc
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        agent.decay_epsilon()
        
        rewards_per_episode.append(total_reward)
        timestep_history.append(steps)
        
        if state[0] >= 0.5:
            solved_episodes.append(episode)
            

        if episode % 100 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_steps = np.mean(timestep_history[-100:])
            
            print(f"Run {run_id}, Episode {episode}/{max_episodes} - Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Avg Steps: {avg_steps:.2f}, Epsilon: {agent.epsilon:.4f}, Time: {elapsed_time:.2f}s")
            
            # Reset time
            start_time = time.time()
    
    # Savando na Q-Table
    qt_filename = f"models/qtable_qlearning_mountaincar_run{run_id}.csv"
    agent.save_q_table(qt_filename)
    
    # Salvando curva de aprendizado
    curve_filename = f"results/learning_curve_qlearning_mountaincar_run{run_id}.csv"
    with open(curve_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "CumulativeReward", "Steps"])
        for ep, (r, s) in enumerate(zip(rewards_per_episode, timestep_history), start=1):
            writer.writerow([ep, r, s])
    
    # Taxa de sucesso
    success_rate = len(solved_episodes) / max_episodes * 100
    print(f"Run {run_id} completed. Success rate: {success_rate:.2f}%")
    
    env.close()
    return rewards_per_episode, timestep_history, agent.q_table

def run_multiple_experiments(runs=5, episodes=5000, steps=1000):
    all_rewards = []
    all_steps = []
    
    for run in range(1, runs+1):
        print(f"\n=== Starting Q-Learning Run {run}/{runs} ===")
        rewards, steps, _ = train_agent(run_id=run, max_episodes=episodes, max_steps=steps)
        all_rewards.append(rewards)
        all_steps.append(steps)
    
    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    mean_steps = np.mean(all_steps, axis=0)
    std_steps = np.std(all_steps, axis=0)

    combined_data = pd.DataFrame({
        'Episode': range(1, episodes+1),
        'MeanReward': mean_rewards,
        'StdReward': std_rewards,
        'MeanSteps': mean_steps,
        'StdSteps': std_steps
    })
    combined_data.to_csv('results/qlearning_mountaincar_combined.csv', index=False)
    

    plot_combined_results(combined_data, 'Q-Learning', 'results/qlearning_mountaincar_learning_curve.png')
    
    print("\n=== Q-Learning Experiment Completed ===")
    print(f"Results saved to 'results/qlearning_mountaincar_combined.csv'")
    print(f"Plot saved to 'results/qlearning_mountaincar_learning_curve.png'")

def plot_combined_results(data, algorithm_name, save_path, window=100):
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
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward (Moving Average)')
    plt.title(f'{algorithm_name} Learning Curve - MountainCar')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning for MountainCar")
    parser.add_argument('--runs', type=int, default=5, help="Number of training runs")
    parser.add_argument('--episodes', type=int, default=5000, help="Number of episodes per run")
    parser.add_argument('--steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--render', action='store_true', help="Render the environment")
    args = parser.parse_args()
    

    print(f"Starting Q-Learning experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.render:
        train_agent(run_id=1, max_episodes=args.episodes, max_steps=args.steps, render=True)
    else:
        run_multiple_experiments(runs=args.runs, episodes=args.episodes, steps=args.steps)
    
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")