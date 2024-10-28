import os
import sys
DEBUG = hasattr(sys, 'gettrace') and (sys.gettrace() is not None)
if DEBUG:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from omegaconf.dictconfig import DictConfig
OmegaConf.register_new_resolver("eval", eval)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mpm'))
import mpm as us
from sim.generate.builder import get_scene
from sim.generate.actions import ActionFactory
from sim.generate.ee import get_ee_state
from sim.generate.topology import Topology
from sim.util import dict_str_to_tuple
import taichi as ti
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
import objaverse.xl as oxl  # Add this import
import trimesh
import requests
import time
from sklearn.decomposition import PCA


class GenerationWorkspace:

    def __init__(self, cfg: OmegaConf):
        OmegaConf.resolve(cfg) # resolve .yaml
        self.cfg = dict_str_to_tuple(cfg) # convert to tuple
        
        # Initialize attributes
        self.cam = None
        self.scene = None
        self.entities = None
        self.planner = None
        self.topology = None
        self.goals = None
        self.init_state = None
        self.scene_dir = None
        self.objaverse_annotations = None

        self.use_objaverse = 'entities' in cfg and 'objaverse_object' in cfg.entities

        # Add new attribute
        self.use_random_actions = False

        # Initialize Objaverse
        if self.use_objaverse:
            self.initialize_objaverse()
        else:
            self.scene, self.cam, self.entities, self.planner = get_scene(self.cfg)

        if self.cfg.render:
            ti.set_logging_level(ti.WARN)
            self.font = ImageFont.truetype("Tests/fonts/NotoSans-Regular.ttf", 18)
        # get topology
        if self.cfg.check.horizon > 0:
            self.topology = Topology(self.cfg, self.scene, self.planner, self.entities)

        us.logger.info(f'=== Creating sequence {cfg.scene_id} in {cfg.log.base_dir} ===')

    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir

    def __del__(self):
        if hasattr(self, 'cam') and self.cam is not None:
            self.cam.window.destroy()
        us.us_exit()

    def reset(self):
        # get scene dir

        if self.cfg.log.base_dir is None:
            us.logger.error('No log.base_dir specified. Adapt the config file.')
            sys.exit(-1)
        elif not os.path.exists(self.cfg.log.base_dir):
            us.logger.error(f'log.base_dir {self.cfg.log.base_dir} does not exist. Create it or adapt the config file.')
            sys.exit(-1)
        if self.cfg.scene_id is None:
            us.logger.error('No scene_id specified. Adapt the config file.')
            sys.exit(-1)
        self.scene_dir = os.path.join(self.cfg.log.base_dir, self.cfg.scene_id)
        if os.path.exists(self.scene_dir):
            us.logger.warning(f'Scene directory {self.scene_dir} already exists. Overwriting.')
        # save resolved config, including overrides and evals
        os.makedirs(self.scene_dir, exist_ok=True)
        OmegaConf.save(self.cfg, os.path.join(self.scene_dir, 'config.yaml'))
        
        # reset simulation, ee and its state machine
        #   optional: let the scene settle and get the initial state for future resets
        self.planner.reset()
        if self.init_state is None:
            if not DEBUG:
                # let the scene settle
                us.logger.debug('Settling...')
                for _ in range(self.cfg.warmup_horizon):
                    self.scene.step()
                us.logger.debug('Scene settled.')
            self.init_state = self.scene.get_state()
        self.scene.reset(self.init_state)

        # Add collapse check after initial settling
        if self.use_objaverse:
            if self.check_mesh_collapse():
                raise RuntimeError("Mesh collapsed during initialization")

        # reset topology
        if self.topology is not None:
            self.topology.reset()

    def render(self, step=-1, info_str=''):
        # render state, add id and time, save visualization per subgoal
        frame = Image.fromarray(self.cam.render())
        offset = self.cam.res[0]//8
        ImageDraw.Draw(frame).text((offset, self.cam.res[1] - offset), f"step {step}, {step*self.cfg.sim.step_dt:0.1f}s",
                                   fill='black', anchor='ls', font=self.font)
        if info_str != '':
            ImageDraw.Draw(frame).text((offset, offset), info_str,
                                       fill='black', anchor='lt', font=self.font)
        return frame

    def log(self, step, collision_info, ee_state):
        frame = {
            'step': step,
            't': step * self.scene.sim.step_dt,
            'frame_idx': step//self.cfg.log.n_iter,
        }
        # get particle info
        num_particles = 0
        particles_pos, particles_vel = [], []
        for entity in self.entities:
            if entity.name == 'ground':
                continue
            num_particles += entity.n
            entity_state = entity.get_state()
            particles_pos += [entity_state.pos.detach().cpu().numpy()]
            particles_vel += [entity_state.vel.detach().cpu().numpy()]
        particles_pos = np.concatenate(particles_pos, axis=0)
        particles_vel = np.concatenate(particles_vel, axis=0)
        if self.topology is not None:
            particles_idx = self.topology.particle_graph.entity_indices.cpu().numpy()  # component label
        else:
            particles_idx = np.zeros(num_particles, dtype=us.ITYPE_NP)  # assume single component
        # get ee collision info
        particles_sdf = collision_info[:, 0]  # relative to ee, clipped at 1.0
        particles_colliding = collision_info[:, 2]  # with ee
        
        # compose log frame
        frame['obj'] = {
            'particles': {
                'num': num_particles,
                'pos': particles_pos,
                'vel': particles_vel,
                'idx': particles_idx,
                'sdf': particles_sdf,
                'colliding': particles_colliding,
            },
        }
        if self.topology is not None:
            frame['topology'] = {
                'components': self.topology.get_num_components(),
                'genus': self.topology.get_genus_per_component(),
            }
        frame['ee'] = get_ee_state(self.planner.ee, ee_state)

        return frame

    def check_and_log(self, step):
        if (step*self.cfg.sim.step_dt) % 1 < self.cfg.sim.step_dt:
            us.logger.info(f'current sim keyframe {step//self.cfg.log.n_iter:05d} - elapsed sim time {step*self.cfg.sim.step_dt:0.1f}s')

        # only log simulation keyframes (i.e., every n_iter steps)
        if step % self.cfg.log.n_iter != 0:
            return None, None
        
        # get current collision info -- contains sdf, influence, is colliding
        collision_info = self.scene.sim.collision_info()
        num_colliding = int(collision_info[:, 2].sum())
        if num_colliding > 0 and step == 0:
            us.logger.error(f'ee in collision at init ({num_colliding} particles in collision) - check ee opening and/or object scale')
            sys.exit(-1)
        # render keyframe
        if not DEBUG and self.cfg.render:
            info_str = f'frame {step//self.cfg.log.n_iter}'
            if self.topology is not None:
                info_str = f'{info_str} - {str(self.topology)}'
            keyframe = self.render(step, info_str=info_str)
        else:
            keyframe = None
        # check topology in keyframe
        if num_colliding > 0 and self.topology is not None:
            self.topology.check()

        # log keyframe
        cur_ee_state = self.planner.ee.get_state()
        log = self.log(step, collision_info, cur_ee_state)

        return keyframe, log

    def run(self):
        self.reset()
        
        # Check for random grasp actions once at the start
        self.use_random_actions = any(action == 'random_grasp' for action in self.cfg.actions)
        original_actions = self.cfg.actions.copy() if self.use_random_actions else None
        
        for grasp_attempt in range(self.cfg.num_grasps):
            logs = []
            frames = []
            
            # Restore original actions and generate new random grasps if needed
            if self.use_random_actions:
                self.cfg.actions = original_actions.copy()
                self.generate_random_grasp_params()
            
            self.goals = ActionFactory.get_goals(self.cfg.actions)
            
            self.reset()
            
            for step in range(self.cfg.max_horizon):
                if self.planner.step(self.goals):
                    break
                self.scene.step()

                keyframe, log = self.check_and_log(step)
                if keyframe is not None:
                    frames.append(keyframe)
                if log is not None:
                    logs.append(log)
            
            us.logger.info(f'=== Completed grasp attempt {grasp_attempt + 1}/{self.cfg.num_grasps} ===')

            # Save logs
            log_path = os.path.join(self.scene_dir, f"log_grasp_{grasp_attempt + 1}.pkl")
            pickle.dump(logs, open(log_path, 'wb'))

            # Save state data with actions
            state_data = {
                'initial_state': logs[0] if logs else None,
                'final_state': logs[-1] if logs else None,
                'actions': dict(self.cfg.actions)  # Add actions to the state data
            }
            state_data_path = os.path.join(self.scene_dir, f"state_data_grasp_{grasp_attempt + 1}.pkl")
            with open(state_data_path, 'wb') as f:
                pickle.dump(state_data, f)

            # Save visualizations
            if not DEBUG and self.cfg.render and frames:
                us.logger.info(f'Saving visualization for grasp attempt {grasp_attempt + 1}.')
                gif_path = os.path.join(self.scene_dir, f"visualization_grasp_{grasp_attempt + 1}.gif")
                frames[0].save(gif_path, format="GIF", append_images=frames[1:], save_all=True, loop=0,
                               duration=int(self.cfg.log.dt*1000))
                
                # Save initial and final frame screenshots
                frames[0].save(os.path.join(self.scene_dir, f"initial_frame_grasp_{grasp_attempt + 1}.png"))
                frames[-1].save(os.path.join(self.scene_dir, f"final_frame_grasp_{grasp_attempt + 1}.png"))

        # Save the object's OBJ file to the scene directory
        if self.use_objaverse:
            obj_source = os.path.join(os.path.dirname(__file__), 'mpm', 'assets', 'meshes', 'objaverse', 'objaverse_object.obj')
            obj_dest = os.path.join(self.scene_dir, 'object.obj')
            import shutil
            shutil.copy2(obj_source, obj_dest)
            
        us.logger.info(f'=== Done simulating all grasps for sequence {self.cfg.scene_id} ===')

    def generate_random_grasp_params(self):
        # Ensure randomness by reseeding the random number generator
        np.random.seed()  # Remove any fixed seed to ensure different results each time

        # Get the number of actions to generate
        num_actions = self.cfg.actions.random_grasp.number_of_actions_per_grasp
        
        # Create a list to store multiple actions
        action_sequence = []
        
        # Generate the specified number of random grasps
        for i in range(num_actions):
            random_pose = self.generate_random_pose()
            
            # Create a new grasp action
            grasp_action = {
                'wait': 0,
                'type': 'grasp',
                'from_offset': np.random.uniform(-0.01, 0.01, 3).tolist(),
                'to_pos': random_pose[:3].tolist(),
                'to_quat': random_pose[3:].tolist(),
                'init_d': float(np.random.uniform(0.4, 0.7)),
                'close_d': float(np.random.uniform(0.0, 0.1))
            }
            
            action_sequence.append(grasp_action)
        
        # Replace the single random_grasp in cfg.actions with the sequence
        # First, remove the original random_grasp
        actions_dict = dict(self.cfg.actions)
        actions_dict.pop('random_grasp', None)
        
        # Add the sequence of actions with unique names
        for i, action in enumerate(action_sequence):
            actions_dict[f'random_grasp_{i+1}'] = action
        
        # Update the config
        self.cfg.actions = actions_dict

    def generate_random_pose(self):

        np.random.seed()  # Remove any fixed seed to ensure different results each time

        # Random position within a reasonable range
        pos = np.random.uniform(-0.01, 0.01, 3)
        
        # Random orientation using uniformly sampled quaternions
        u1, u2, u3 = np.random.random(3)
        
        # Use the "subgroup algorithm" to generate uniform quaternions
        sqrt1_u1 = np.sqrt(1 - u1)
        sqrtu1 = np.sqrt(u1)
        quat = np.array([
            sqrt1_u1 * np.sin(2 * np.pi * u2),
            sqrt1_u1 * np.cos(2 * np.pi * u2),
            sqrtu1 * np.sin(2 * np.pi * u3),
            sqrtu1 * np.cos(2 * np.pi * u3)
        ])
        
        return np.concatenate([pos, quat])

    def initialize_objaverse(self):
        # Get annotations from Objaverse
        self.objaverse_annotations = oxl.get_annotations()
        
        # Create a directory for Objaverse meshes if it doesn't exist
        meshes_dir = os.path.join(os.path.dirname(__file__), 'mpm', 'assets', 'meshes', 'objaverse')
        os.makedirs(meshes_dir, exist_ok=True)
        
        # List of file extensions that Trimesh can process
        processable_extensions = ['.obj', '.stl', '.ply', '.glb', '.gltf', '.dae', '.off']

        while True:
            try:
                while True:
                    sampled_object = self.objaverse_annotations.sample(1)
                    object_url = sampled_object.iloc[0]['fileIdentifier']
                    print(f"Trying object_url: {object_url}")

                    # Check if the URL is from GitHub and has a processable extension
                    if 'github.com' in object_url:
                        file_extension = os.path.splitext(object_url)[1].lower()
                        if file_extension in processable_extensions:
                            # Check if the file name starts with "scene."
                            file_name = os.path.basename(object_url).lower()
                            if not file_name.startswith("scene."):
                                break
                            else:
                                print("Skipping 'scene.*' file, sampling again...")
                        else:
                            print(f"Skipping file with unsupported extension: {file_extension}")
                    else:
                        print("Skipping non-GitHub URL, sampling again...")

                # Convert GitHub URL to raw content URL
                if 'github.com' in object_url:
                    object_url = object_url.replace('github.com', 'raw.githubusercontent.com')
                    object_url = object_url.replace('/blob/', '/')
                
                print(f"Downloading from: {object_url}")
                
                # Generate a unique filename for the downloaded object
                object_id = os.path.basename(object_url).split('?')[0]  # Remove query parameters if any
                file_extension = os.path.splitext(object_id)[1]
                if not file_extension:
                    file_extension = '.glb'  # Default to .glb if no extension is present
                
                # Use a consistent filename for the downloaded object
                download_path = os.path.join(meshes_dir, f"objaverse_object{file_extension}")
                
                # Use requests to download the file
                response = requests.get(object_url, timeout=30)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                with open(download_path, 'wb') as f:
                    f.write(response.content)
                
                # If we reach here, the download was successful
                break  # Exit the outer while loop
            
            except Exception as e:
                print(f"Error downloading object: {e}")
                time.sleep(1)  # Wait a bit before trying again
        
        print(f"Successfully downloaded object from: {object_url}")

        # Convert the downloaded file to OBJ format
        obj_path = os.path.join(meshes_dir, "objaverse_object.obj")
        self.convert_to_obj(download_path, obj_path)
        
        # Try to load the mesh and check its properties
        try:
            mesh = trimesh.load(obj_path)
            
            # Check if it's a PointCloud or lacks necessary attributes
            if isinstance(mesh, trimesh.PointCloud) or not hasattr(mesh, 'center_mass') or not hasattr(mesh, 'faces'):
                print("Object is a PointCloud or lacks necessary attributes. Sampling a new object...")
                self.initialize_objaverse()  # Go back to the start of the outer while loop
                return
            
            # Check if the object is very thin using PCA
            vertices = mesh.vertices - mesh.center_mass
            pca = PCA(n_components=3)
            pca.fit(vertices)
            
            # Get the ratio of the smallest variance to the largest
            variance_ratio = pca.explained_variance_[2] / pca.explained_variance_[0]
            
            # You can adjust this threshold as needed
            if variance_ratio < 0.01:  # Object is considered thin if the ratio is less than 1%
                print("Object is too thin. Sampling a new object...")
                self.initialize_objaverse()
                return

        except Exception as e:
            print(f"Error processing mesh: {e}. Sampling a new object...")
            self.initialize_objaverse()
            return
        # Remove the original file if it was converted to OBJ
        if download_path != obj_path:
            os.remove(download_path)
        
        # Update the objaverse.yaml configuration
        objaverse_config = self.cfg.entities.objaverse_object
        objaverse_config.geom.file = "sim/mpm/assets/meshes/objaverse/objaverse_object.obj"
        objaverse_config.objaverse_id = object_id

        try:
            scene, cam, entities, planner = get_scene(self.cfg)
            self.scene = scene
            self.cam = cam
            self.entities = entities
            self.planner = planner
        except Exception as e:
            print(f"Error initializing scene with Objaverse object: {e}")
            self.initialize_objaverse()
            return
        
        print(f"Using Objaverse object: {object_url}")

    # need bpy for fbx
    def convert_to_obj(self, input_path, output_path):

        # Load the mesh
        mesh = trimesh.load(input_path)

        # If the mesh is a scene, export all meshes
        if isinstance(mesh, trimesh.Scene):
            combined_mesh = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in mesh.geometry.values()
            ])
            combined_mesh.export(output_path)
        else:
            # Export the mesh as OBJ
            mesh.export(output_path)

    def check_mesh_collapse(self):
        """Check if the mesh has collapsed by comparing its size before and after initial simulation."""
        us.logger.info("Checking mesh collapse...")
        
        # Get initial bounding box
        initial_positions = []
        for entity in self.entities:
            if entity.name == 'objaverse_object':
                initial_positions = entity.get_state().pos.detach().cpu().numpy()
                break
        
        if len(initial_positions) == 0:
            return False

        initial_bbox = np.ptp(initial_positions, axis=0)  # Range of points in each dimension
        initial_volume = np.prod(initial_bbox)

        # Simulate a few steps
        for _ in range(10):  # Adjust number of steps as needed
            self.scene.step()

        # Get new bounding box
        current_positions = []
        for entity in self.entities:
            if entity.name == 'objaverse_object':
                current_positions = entity.get_state().pos.detach().cpu().numpy()
                break

        current_bbox = np.ptp(current_positions, axis=0)
        current_volume = np.prod(current_bbox)

        # Check if volume has significantly decreased
        volume_ratio = current_volume / initial_volume
        return volume_ratio < 0.5  # Adjust threshold as needed

@hydra.main(
    version_base=None,
    config_path='./generate/config', 
    config_name='template')
def main(cfg):
    workspace = GenerationWorkspace(cfg)
    workspace.run()


if  __name__== '__main__':
    main()

