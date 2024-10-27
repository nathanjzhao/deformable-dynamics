import os
import trimesh
import numpy as np
from .misc import *
import urllib.parse
import logging
import requests
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_raw_mesh_path(file):
    us.logger.debug(f"get_raw_mesh_path called with file: {file}")
    
    # Check if the file string contains "github.com"
    if "github.com" in file:
        us.logger.debug("File appears to be a GitHub URL")
        # Extract the actual URL
        url = file[file.index("http"):]
        
        # Convert GitHub URL to raw content URL
        url_parts = url.split("/")
        if "blob" in url_parts:
            blob_index = url_parts.index("blob")
            url_parts[blob_index] = "raw"
            raw_url = "/".join(url_parts)
        else:
            raw_url = url
        
        us.logger.debug(f"Raw GitHub URL: {raw_url}")
        
        # Parse the URL to get the filename
        parsed_url = urllib.parse.urlparse(raw_url)
        base_name = os.path.basename(parsed_url.path)
        
        # Define the local path where we'll save the downloaded file
        local_path = os.path.join(get_src_dir(), 'assets', 'meshes', 'raw', base_name)
        
        # Download the file if it doesn't exist locally
        if not os.path.exists(local_path):
            us.logger.debug(f"Downloading file from {raw_url}")
            try:
                response = requests.get(raw_url)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                us.logger.error(f"Error downloading file: {e}")
                return None
        
        file = local_path  # Use the local path for further processing
    
    us.logger.debug(f"Final file path: {file}")
    
    # Get the file extension
    _, ext = os.path.splitext(file)
    
    # If it's already an .obj file, just return the path
    if ext.lower() == '.obj':
        return file
    
    # For other formats, we'll convert to .obj
    base_name = os.path.basename(file)
    obj_file = f"{os.path.splitext(base_name)[0]}.obj"
    obj_path = os.path.join(get_src_dir(), 'assets', 'meshes', 'raw', obj_file)
    
    # If the .obj file doesn't exist, create it
    if not os.path.exists(obj_path):
        us.logger.debug(f"Converting {file} to {obj_path}")
        try:
            # Load the mesh
            mesh = load_mesh(file)
            if mesh is None:
                us.logger.error(f"Failed to load mesh from {file}")
                return None
            
            # Export as .obj
            mesh.export(obj_path)
        except Exception as e:
            us.logger.error(f"Error processing file: {e}")
            return None
    
    return obj_path

def get_processed_mesh_path(file, file_vis):
    assert file.endswith('.obj') and file_vis.endswith('.obj')
    processed_file      = f"{file.replace('.obj', '')}-{file_vis.replace('.obj', '')}.obj"
    processed_file_path = os.path.join(get_src_dir(), 'assets', 'meshes', 'processed', processed_file)
    return processed_file_path

def get_processed_sdf_path(file, sdf_res):
    assert file.endswith('.obj')
    processed_sdf = f"{file.replace('.obj', '')}-{sdf_res}.sdf"
    processed_sdf_path = os.path.join(get_src_dir(), 'assets', 'meshes', 'processed', processed_sdf)
    return processed_sdf_path

def get_voxelized_mesh_path(file, voxelize_res):
    base_name = os.path.basename(file)
    vox_file = f"{os.path.splitext(base_name)[0]}-{voxelize_res}.pkl"
    voxelized_dir = os.path.join(get_src_dir(), 'assets', 'meshes', 'voxelized')
    vox_path = os.path.join(voxelized_dir, vox_file)

    # Create the directory if it doesn't exist
    os.makedirs(voxelized_dir, exist_ok=True)

    # Load the mesh
    raw_mesh = load_mesh(file)
    
    if raw_mesh is None:
        us.logger.error(f"Failed to load mesh from {file} for voxelization")
        return None
    
    # Voxelize the mesh
    voxelized_matrix = voxelize_mesh(raw_mesh, voxelize_res)
    
    if voxelized_matrix is None:
        us.logger.error(f"Failed to voxelize mesh: {file}")
        return None
    
    # Save the voxelized mesh as a pickle file
    with open(vox_path, 'wb') as f:
        pickle.dump(voxelized_matrix, f)
    us.logger.debug(f"Voxelized mesh saved to {vox_path}")

    return vox_path

def load_mesh(file):
    try:
        # Try to determine the file type
        file_type = trimesh.util.split_extension(file)
        
        if file_type is None or file_type == 'nonetype':
            print(f"Warning: Unable to determine file type for {file}")
            # Try to load it anyway, trimesh might be able to infer the type
            return trimesh.load(file, force='mesh', skip_texture=True)
        
        # If file type is determined, proceed with loading
        return trimesh.load(file, file_type=file_type, force='mesh', skip_texture=True)
    
    except Exception as e:
        print(f"Error loading mesh from {file}: {str(e)}")
        print(f"File exists: {os.path.exists(file)}")
        print(f"File size: {os.path.getsize(file) if os.path.exists(file) else 'N/A'}")
        return None

def normalize_mesh(mesh, mesh_actual=None):
    '''
    Normalize mesh_dict to [-0.5, 0.5] using size of mesh_dict_actual.
    '''
    if mesh_actual is None:
        mesh_actual = mesh

    scale  = (mesh_actual.vertices.max(0) - mesh_actual.vertices.min(0)).max()
    center = (mesh_actual.vertices.max(0) + mesh_actual.vertices.min(0))/2

    normalized_mesh = mesh.copy()
    normalized_mesh.vertices -= center
    normalized_mesh.vertices /= scale
    return normalized_mesh

def scale_mesh(mesh, scale):
    scale = np.array(scale)
    return trimesh.Trimesh(
        vertices = mesh.vertices * scale,
        faces    = mesh.faces,
    )

def cleanup_mesh(mesh):
    '''
    Retain only mesh's vertices, faces, and normals.
    '''

    return trimesh.Trimesh(
        vertices       = mesh.vertices,
        faces          = mesh.faces,
        vertex_normals = mesh.vertex_normals,
        face_normals   = mesh.face_normals,
    )

def voxelize_mesh(mesh, voxelize_res):
    try:
        # Normalize and cleanup the mesh
        normalized_mesh = cleanup_mesh(normalize_mesh(mesh))
        
        # Voxelize the normalized mesh
        voxel_grid = normalized_mesh.voxelized(pitch=1.0/voxelize_res).fill()
        
        # Convert to numpy array
        voxel_matrix = voxel_grid.matrix
        
        # Ensure the voxel matrix is 3D
        if voxel_matrix.ndim < 3:
            us.logger.warning(f"Voxel matrix is {voxel_matrix.ndim}D, expanding to 3D")
            voxel_matrix = np.expand_dims(voxel_matrix, axis=2)
        
        us.logger.debug(f"Voxelized mesh shape: {voxel_matrix.shape}")
        
        # Visualize the voxelized mesh
        visualize_voxelized_mesh(voxel_matrix, voxelize_res, mesh.metadata.get('file_path', 'unknown'))
        
        return voxel_matrix
    except Exception as e:
        us.logger.error(f"Error during voxelization: {str(e)}")
        raise e

def visualize_voxelized_mesh(voxel_matrix, voxelize_res, original_file_path):
    try:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x, y, z = np.where(voxel_matrix == 1)
        ax.scatter(x, y, z, c='r', marker='s')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Voxelized Mesh (Resolution: {voxelize_res})')
        
        # Generate the output filename
        base_name = os.path.basename(original_file_path)
        output_filename = f"{os.path.splitext(base_name)[0]}_voxelized_{voxelize_res}.png"
        output_dir = os.path.join(get_src_dir(), 'assets', 'meshes', 'voxelized')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        
        us.logger.info(f"Voxelized mesh visualization saved to {output_path}")
    except Exception as e:
        us.logger.error(f"Error visualizing voxelized mesh: {str(e)}")
