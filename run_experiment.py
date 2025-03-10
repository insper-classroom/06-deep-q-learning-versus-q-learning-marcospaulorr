import subprocess
import os
import time
import argparse
from datetime import datetime

def create_directories():
    """
    Create necessary directories for the experiment
    """
    directories = ['results', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def run_command(command, description):
    """
    Run a shell command and log the output
    """
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    process = subprocess.Popen(command, shell=True)
    process.wait()
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Completed: {description}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"{'='*50}\n")
    
    return process.returncode == 0

def run_experiment(episodes=1000, runs=5, steps=1000, validate_episodes=100):
    """
    Run the full experiment pipeline
    """
    # Create directories
    create_directories()
    
    # Start time
    start_time = time.time()
    overall_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting experiment at {overall_start}")
    
    # 1. Train Q-Learning agent
    success = run_command(
        f"python train_mountaincar_qlearning.py --episodes {episodes} --runs {runs} --steps {steps}",
        "Q-Learning Training"
    )
    if not success:
        print("Error during Q-Learning training. Stopping experiment.")
        return False
    
    # 2. Train Deep Q-Learning agent
    success = run_command(
        f"python train_mountaincar_deepqlearning.py --episodes {episodes} --runs {runs} --steps {steps}",
        "Deep Q-Learning Training"
    )
    if not success:
        print("Error during Deep Q-Learning training. Stopping experiment.")
        return False
    
    # 3. Validate Q-Learning agent
    success = run_command(
        f"python validate_mountaincar_qlearning.py --episodes {validate_episodes} --qtable models/qtable_qlearning_mountaincar_run1.csv",
        "Q-Learning Validation"
    )
    if not success:
        print("Error during Q-Learning validation. Continuing anyway...")
    
    # 4. Validate Deep Q-Learning agent
    success = run_command(
        f"python validate_mountaincar_deepqlearning.py --episodes {validate_episodes} --model models/deepq_mountaincar_model_run1.keras",
        "Deep Q-Learning Validation"
    )
    if not success:
        print("Error during Deep Q-Learning validation. Continuing anyway...")
    
    # 5. Generate comparison plots
    success = run_command(
        "python plot_comparison.py",
        "Generating Comparison Plots"
    )
    if not success:
        print("Error during plot generation. Continuing anyway...")
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*50}")
    print(f"Experiment completed!")
    print(f"Started at: {overall_start}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"{'='*50}\n")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full MountainCar experiment pipeline")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes for training")
    parser.add_argument('--runs', type=int, default=5, help="Number of training runs per algorithm")
    parser.add_argument('--steps', type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument('--validate_episodes', type=int, default=100, help="Number of episodes for validation")
    args = parser.parse_args()
    
    run_experiment(
        episodes=args.episodes,
        runs=args.runs,
        steps=args.steps,
        validate_episodes=args.validate_episodes
    )