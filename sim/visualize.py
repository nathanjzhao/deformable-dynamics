import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_trajectory(log_path):
    # Load the pickle file
    with open(log_path, 'rb') as f:
        logs = pickle.load(f)
    
    # Get the directory of the log file
    log_dir = os.path.dirname(log_path)
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize each frame
    for i, frame in enumerate(logs):
        ax.clear()
        
        # Get particle positions and collision states
        particles = frame['obj']['particles']
        positions = particles['pos']  # Shape should be (n_particles, 3)
        colliding = particles['colliding']  # Boolean array for particles colliding with ee
        
        # Ensure colliding is a boolean array
        colliding = np.array(colliding, dtype=bool)
        
        # Plot non-colliding particles in blue, colliding in red
        ax.scatter(positions[~colliding, 0], positions[~colliding, 1], positions[~colliding, 2], 
                  c='blue', alpha=0.6, label='Non-colliding')
        ax.scatter(positions[colliding, 0], positions[colliding, 1], positions[colliding, 2], 
                  c='red', alpha=0.6, label='Colliding')
        
        # Plot end effector position if available
        if 'ee' in frame:
            ee_pos = frame['ee']['pos']
            ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], c='green', s=100, label='End effector')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time: {frame["t"]:.2f}s')
        ax.legend()
        
        # Save the figure
        save_path = os.path.join(log_dir, f'frame_{i:04d}.png')
        plt.savefig(save_path)
        print(f"Saved frame {i} to {save_path}")
        
        plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    # Usage
    log_path = "data/new/log.pkl"
    visualize_trajectory(log_path)
