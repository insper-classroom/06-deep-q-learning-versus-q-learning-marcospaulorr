import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def moving_average(data, window_size=100):
    """
    Calculate the moving average of a list or array
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, creating it if it doesn't
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_q_table(q_table, filename, shape=None):
    """
    Save a Q-table to a CSV file
    """
    ensure_directory_exists(os.path.dirname(filename))
    
    if shape is not None:
        # Flatten the 3D table to 2D for saving
        rows, cols, actions = shape
        q_table_2d = q_table.reshape(rows * cols, actions)
        np.savetxt(filename, q_table_2d, delimiter=",")
    else:
        np.savetxt(filename, q_table, delimiter=",")

def load_q_table(filename, shape=None):
    """
    Load a Q-table from a CSV file
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Q-table file {filename} not found")
    
    q_table_2d = np.loadtxt(filename, delimiter=",")
    
    if shape is not None:
        return q_table_2d.reshape(shape[0], shape[1], shape[2])
    else:
        return q_table_2d

def plot_learning_curve(data, title, xlabel, ylabel, save_path=None, window=100):
    """
    Plot a learning curve with moving average
    """
    if isinstance(data, list):
        data = np.array(data)
    
    # Apply moving average
    ma_data = moving_average(data, window)
    episodes = np.arange(len(ma_data))
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ma_data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path)
    
    plt.close()

def plot_comparison(data1, data2, label1, label2, title, xlabel, ylabel, save_path=None, window=100):
    """
    Plot a comparison of two learning curves with moving average
    """
    if isinstance(data1, list):
        data1 = np.array(data1)
    if isinstance(data2, list):
        data2 = np.array(data2)
    
    # Apply moving average
    ma_data1 = moving_average(data1, window)
    ma_data2 = moving_average(data2, window)
    
    # Make sure both arrays have the same length by truncating to the shorter one
    min_length = min(len(ma_data1), len(ma_data2))
    ma_data1 = ma_data1[:min_length]
    ma_data2 = ma_data2[:min_length]
    episodes = np.arange(min_length)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ma_data1, label=label1)
    plt.plot(episodes, ma_data2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path)
    
    plt.close()