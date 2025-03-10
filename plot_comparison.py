import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

def load_data(filename):
    """
    Load data from a CSV file
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    
    return pd.read_csv(filename)

def plot_learning_curves(q_file, dq_file, window=100, save_path="results/comparison_learning_curves.png"):
    """
    Plot learning curves for Q-Learning and Deep Q-Learning
    """
    # Load data
    q_data = load_data(q_file)
    dq_data = load_data(dq_file)
    
    # Apply moving average
    q_data['MeanReward_MA'] = q_data['MeanReward'].rolling(window=window).mean()
    dq_data['MeanReward_MA'] = dq_data['MeanReward'].rolling(window=window).mean()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot Q-Learning
    plt.plot(q_data['Episode'], q_data['MeanReward_MA'], label='Q-Learning', color='blue')
    plt.fill_between(
        q_data['Episode'],
        q_data['MeanReward_MA'] - q_data['StdReward'],
        q_data['MeanReward_MA'] + q_data['StdReward'],
        alpha=0.2,
        color='blue'
    )
    
    # Plot Deep Q-Learning
    plt.plot(dq_data['Episode'], dq_data['MeanReward_MA'], label='Deep Q-Learning', color='red')
    plt.fill_between(
        dq_data['Episode'],
        dq_data['MeanReward_MA'] - dq_data['StdReward'],
        dq_data['MeanReward_MA'] + dq_data['StdReward'],
        alpha=0.2,
        color='red'
    )
    
    # Add a horizontal line for "solved" threshold
    plt.axhline(y=-110, color='green', linestyle='--', label='Solved Threshold (-110)')
    
    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Moving Average)')
    plt.title('MountainCar Learning Curves: Q-Learning vs Deep Q-Learning')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curves comparison saved to {save_path}")

def plot_inference_comparison(q_results, dq_results, save_path="results/comparison_inference.png"):
    """
    Plot inference comparison for Q-Learning and Deep Q-Learning
    """
    # Categories
    categories = ['Success Rate (%)', 'Avg Steps', 'Avg Reward', 'Avg Position']
    
    # Values
    q_values = [
        q_results['success_rate'],
        q_results['avg_steps'],
        abs(q_results['avg_reward']),  # Use absolute value for better visualization
        q_results['avg_position'] * 100  # Multiply by 100 for better scale
    ]
    
    dq_values = [
        dq_results['success_rate'],
        dq_results['avg_steps'],
        abs(dq_results['avg_reward']),  # Use absolute value for better visualization
        dq_results['avg_position'] * 100  # Multiply by 100 for better scale
    ]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.3
    
    # Set position of bar on X axis
    r1 = np.arange(len(categories))
    r2 = [x + barWidth for x in r1]
    
    # Make the plot
    plt.bar(r1, q_values, width=barWidth, edgecolor='grey', label='Q-Learning')
    plt.bar(r2, dq_values, width=barWidth, edgecolor='grey', label='Deep Q-Learning')
    
    # Add labels
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Values')
    plt.xticks([r + barWidth/2 for r in range(len(categories))], categories)
    
    # Create legend & title
    plt.title('MountainCar Inference Comparison: Q-Learning vs Deep Q-Learning')
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(q_values):
        plt.text(r1[i] - 0.05, v + 0.1, f"{v:.1f}", color='blue', fontweight='bold')
    
    for i, v in enumerate(dq_values):
        plt.text(r2[i] - 0.05, v + 0.1, f"{v:.1f}", color='red', fontweight='bold')
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Inference comparison saved to {save_path}")

def extract_validation_results(filename):
    """
    Extract validation results from a text file
    """
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Success Rate:" in line:
                results['success_rate'] = float(line.split(":")[1].strip().replace("%", ""))
            elif "Average Reward:" in line:
                results['avg_reward'] = float(line.split(":")[1].strip())
            elif "Average Steps:" in line:
                results['avg_steps'] = float(line.split(":")[1].strip())
            elif "Average Position Reached:" in line:
                results['avg_position'] = float(line.split(":")[1].strip())
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison between Q-Learning and Deep Q-Learning")
    parser.add_argument('--q_data', type=str, default="results/qlearning_mountaincar_combined.csv", 
                        help="Path to Q-Learning combined data file")
    parser.add_argument('--dq_data', type=str, default="results/deepq_mountaincar_combined.csv", 
                        help="Path to Deep Q-Learning combined data file")
    parser.add_argument('--q_validation', type=str, default="models/qtable_qlearning_mountaincar_run1_validation_results.txt", 
                        help="Path to Q-Learning validation results file")
    parser.add_argument('--dq_validation', type=str, default="models/deepq_mountaincar_model_run1_validation_results.txt", 
                        help="Path to Deep Q-Learning validation results file")
    parser.add_argument('--window', type=int, default=100, help="Window size for moving average")
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Plot learning curves
    plot_learning_curves(args.q_data, args.dq_data, args.window)
    
    # Extract validation results
    q_results = extract_validation_results(args.q_validation)
    dq_results = extract_validation_results(args.dq_validation)
    
    # Plot inference comparison
    plot_inference_comparison(q_results, dq_results)