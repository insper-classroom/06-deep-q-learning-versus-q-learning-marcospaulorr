import gymnasium as gym
import numpy as np
import argparse
import tensorflow as tf
import time
import os

def validate_agent(model_file, episodes=100, max_steps=1000, render=False):
    """
    Validate a trained Deep Q-Learning agent
    """
    # Check if model file exists
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found")
    
    # Load the model
    model = tf.keras.models.load_model(model_file)
    
    # Create environment
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    
    # Statistics variables
    rewards_list = []
    steps_list = []
    success_count = 0
    position_reached = []
    
    print(f"Starting validation with {episodes} episodes...")
    
    for episode in range(1, episodes+1):
        state, _ = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        episode_reward = 0
        steps = 0
        done = False
        
        for step in range(max_steps):
            # Select action (greedy policy)
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Break if done
            if done:
                break
        
        # Extract the final position (x-coordinate)
        final_position = state[0][0]
        
        # Check if solved (reached the flag)
        if final_position >= 0.5:
            success_count += 1
        
        # Record stats
        rewards_list.append(episode_reward)
        steps_list.append(steps)
        position_reached.append(final_position)
        
        # Print progress
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode}/{episodes} - Reward: {episode_reward:.2f}, Steps: {steps}, "
                  f"Position: {final_position:.4f}")
    
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
    
    # Save results to file
    results_file = model_file.replace(".keras", "_validation_results.txt")
    with open(results_file, "w") as f:
        f.write("Validation Results:\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")
        f.write(f"Average Steps: {avg_steps:.2f}\n")
        f.write(f"Average Position Reached: {avg_position:.4f}\n")
    
    env.close()
    return success_rate, avg_reward, avg_steps, avg_position

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Deep Q-Learning agent for MountainCar")
    parser.add_argument('--model', type=str, required=False, 
                        default="models/deepq_mountaincar_model_run1.keras",
                        help="Path to model file")
    parser.add_argument('--episodes', type=int, default=100, help="Number of episodes for validation")
    parser.add_argument('--steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--render', action='store_true', help="Render the environment")
    args = parser.parse_args()
    
    validate_agent(args.model, args.episodes, args.steps, args.render)