import tensorflow as tf
import numpy as np
import glob
import pickle
import os

def create_tf_example(trajectory_data):
    """Convert a trajectory into a tf.Example."""
    
    # Extract position sequences from logs
    # Assuming each log has particle positions in log['obj']['particles']['pos']
    positions = []
    for log in trajectory_data:
        pos = log['obj']['particles']['pos']
        positions.append(pos)
    
    # Convert to numpy array with shape [sequence_length, num_particles, 3]
    positions = np.array(positions)
    
    # Create particle types - if you don't have specific types, you could use all same type
    num_particles = positions.shape[1]
    particle_types = np.zeros(num_particles, dtype=np.int64)  # Default type 0
    
    # Optional: Extract global context if you have it
    # global_context = ...

    # Create tf.Example
    feature = {
        'position': tf.train.Feature(
            float_list=tf.train.FloatList(value=positions.reshape(-1))),
        'particle_type': tf.train.Feature(
            int64_list=tf.train.Int64List(value=particle_types)),
        'position_shape': tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(positions.shape))),
    }
    
    # Add optional global context if available
    # if global_context is not None:
    #     feature['step_context'] = tf.train.Feature(
    #         float_list=tf.train.FloatList(value=global_context.reshape(-1)))
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def convert_logs_to_tfrecord(input_pattern, output_path):
    """Convert all log files matching input_pattern to a single TFRecord file."""
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
    
    writer = tf.io.TFRecordWriter(output_path)
    
    # Get all log files
    log_files = glob.glob(input_pattern)
    
    for log_file in log_files:
        # Load pickle file
        with open(log_file, 'rb') as f:
            trajectory_data = pickle.load(f)
        
        # Convert to tf.Example
        tf_example = create_tf_example(trajectory_data)
        
        # Write to TFRecord file
        writer.write(tf_example.SerializeToString())
    
    writer.close()

def create_metadata(log_files, output_path):
    """Create metadata.json required by the training code."""
    
    # Calculate statistics across all trajectories
    all_positions = []
    all_velocities = []
    
    for log_file in log_files:
        with open(log_file, 'rb') as f:
            trajectory_data = pickle.load(f)
            
        for log in trajectory_data:
            pos = log['obj']['particles']['pos']
            vel = log['obj']['particles']['vel']
            all_positions.append(pos)
            all_velocities.append(vel)
    
    all_positions = np.concatenate(all_positions, axis=0)
    all_velocities = np.concatenate(all_velocities, axis=0)
    
    # Calculate statistics
    metadata = {
        'dim': 3,  # 3D positions
        'sequence_length': len(trajectory_data),  # number of steps in trajectory
        'default_connectivity_radius': 0.05,  # adjust based on your simulation
        'bounds': [[-1, 1], [-1, 1], [-1, 1]],  # adjust based on your workspace
        
        # Statistics for normalization
        'vel_mean': np.mean(all_velocities, axis=0).tolist(),
        'vel_std': np.std(all_velocities, axis=0).tolist(),
        'acc_mean': [0, 0, 0],  # calculate if you have acceleration data
        'acc_std': [1, 1, 1],   # calculate if you have acceleration data
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
  # Usage example:
  input_pattern = '../data/scene_*/log_grasp_*.pkl'  # Matches all .pkl files in logs subdirectories of scene_ directories
  output_dir = 'formatted_data/'

  # Create train/valid/test splits
  train_output = f'{output_dir}/train.tfrecord'
  valid_output = f'{output_dir}/valid.tfrecord'
  test_output = f'{output_dir}/test.tfrecord'

  # Convert logs to TFRecord format
  convert_logs_to_tfrecord(input_pattern, train_output)

  # Create metadata file
  create_metadata(glob.glob(input_pattern), f'{output_dir}/metadata.json')
