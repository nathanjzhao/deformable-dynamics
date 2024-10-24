import os
import uuid
import numpy as np
import mpm as us
import taichi as ti
import pickle as pkl
from mpm.utils.mesh import *
from scipy.spatial.transform import Rotation
from mpm.utils.repr import _repr, _repr_uuid

# Global variable for target particle count
TARGET_PARTICLE_COUNT = 4163

@ti.data_oriented
class ParticleEntity:
    '''
    Base class for particle-based entity.
    '''
    def __init__(self, scene, solver, material, geom, surface_options, particle_diameter, idx_offset=0):
        self.scene             = scene
        self.sim               = scene.sim
        self.solver            = solver
        self.material          = material
        self.particle_diameter = particle_diameter
        self.idx_offset        = idx_offset
        self.id                = str(uuid.uuid4())
        self.geom              = geom
        self.surface_options   = surface_options
        self.step_global_added = None
        self.name              = geom.name

        self.sample()

        self.init_tgt_vars()

        self.active = False # This attribute is only used in forward pass. It should NOT be used during backward pass.

    def init_filling(self):
        self.filling = self.material.filling

    def init_tgt_keys(self):
        self._tgt_keys = ['vel', 'pos', 'act', 'vel_masked', 'col']

    def add_to_solver(self):
        self.add_to_solver_kernel(self.sim.cur_substep_local, self.init_positions, np.array(self.surface_options.color, dtype=us.FTYPE_NP))
        self.active = True

    def sample(self):
        self.init_filling()
        
        if isinstance(self.geom, us.options.geoms.Cube):
            positions = self.sample_cube()
        elif isinstance(self.geom, us.options.geoms.Cylinder):
            positions = self.sample_cylinder()
        elif isinstance(self.geom, us.options.geoms.Supertoroid):
            positions = self.sample_supertoroid()
        elif isinstance(self.geom, us.options.geoms.Mesh):
            positions = self.sample_mesh()
        elif isinstance(self.geom, us.options.geoms.Particles):
            positions = self.sample_particles()
        else:
            us.raise_exception(f'Unsupported geom: {type(self.geom)}.')

        self.instantiate(positions.astype(us.FTYPE_NP))

    def init_tgt_vars(self):
        # temp variable to store targets for next step
        self._tgt = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None

    def save_ckpt(self):
        pass

    def _sample_cube(self, lower, upper, filling, n_particles):
        size = upper - lower
        if filling == 'random':
            positions = np.random.uniform(low=lower, high=upper, size=(n_particles, 3))
        elif filling == 'grid':
            n_per_dim = int(np.ceil(n_particles**(1/3)))
            x = np.linspace(lower[0], upper[0], n_per_dim)
            y = np.linspace(lower[1], upper[1], n_per_dim)
            z = np.linspace(lower[2], upper[2], n_per_dim)
            positions = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1).reshape((-1, 3))
            positions = positions[:n_particles]
        elif filling == 'jittered':
            positions = self._sample_cube(lower, upper, 'grid', n_particles)
            max_offset = self.particle_diameter/2  # in any direction
            positions += np.random.uniform(low=-max_offset, high=max_offset, size=positions.shape)
            positions = np.clip(positions, lower, upper)
        else:
            us.raise_exception(f'Unsupported filling type: {filling}.')
        return positions

    def instantiate(self, positions):
        # rotate
        R = Rotation.from_euler('zyx', np.array(self.geom.euler)[::-1], degrees=True).as_matrix()
        particles_COM = positions.mean(0)
        init_positions = (R @ (positions - particles_COM).T).T + particles_COM

        if not init_positions.shape[0] > 0:
            us.raise_exception(f'Entity has zero particles.')

        if not self.solver.boundary.is_inside(init_positions):
            us.raise_exception(f'Entity has particles outside solver boundary. Note that solver boundary is slightly smaller than the specified domain due to safety padding.\n\nCurrent boundary:\n{self.solver.boundary}\n\nEntity to be added:\nmin: {init_positions.min(0)}\nmax: {init_positions.max(0)}\n')

        self.init_positions = torch.tensor(init_positions).cuda().contiguous().to(us.FTYPE_TC)
        self.init_positions_COM_offset = (self.init_positions - torch.tensor(particles_COM).cuda()).contiguous().to(us.FTYPE_TC)

    def compute_n_particles(self, volume):
        return round(volume / self.particle_diameter ** 3)

    def compute_n_particles_1D(self, length):
        return round(length / self.particle_diameter)

    def set_color(self, col):
        '''
        Accepted tensor shape: (4,) or (self.n, 4).
        '''
        self._assert_active()

        col = col.clone()

        if len(col.shape) == 1:
            assert col.shape == (4,)
            self._tgt['col'] = torch.tile(col, [self.n, 1])

        elif len(col.shape) == 2:
            assert col.shape == (self.n, 4)
            self._tgt['col'] = col

        else:
            us.raise_exception('Tensor shape not supported.')

    def set_velocity(self, vel):
        '''
        Accepted tensor shape: (3,) or (self.n, 3).
        '''
        self._assert_active()
        # us.logger.warning('Manally setting particle velocities. This is not recommended and could break gradient flow.')

        vel = vel.clone()

        if len(vel.shape) == 1:
            assert vel.shape == (3,)
            self._tgt['vel'] = torch.tile(vel, [self.n, 1])

        elif len(vel.shape) == 2:
            assert vel.shape == (self.n, 3)
            self._tgt['vel'] = vel

        else:
            us.raise_exception('Tensor shape not supported.')
    
    def set_velocity_masked(self, vel_masked):
        '''
        Accepted tensor shape: (self.n, 4).
        '''
        self._assert_active()
        # us.logger.warning('Manally setting particle velocities. This is not recommended and could break gradient flow.')

        vel_masked = vel_masked.clone()

        if len(vel_masked.shape) == 2:
            assert vel_masked.shape == (self.n, 4)
            self._tgt['vel_masked'] = vel_masked

        else:
            us.raise_exception('Tensor shape not supported.')

    def set_position(self, pos):
        '''
        Accepted tensor shape: (3,) for COM position or (self.n, 3) for particle-wise position.
        When COM position is given, the particles will be restored to the entity's initial shape.
        '''
        self._assert_active()
        # us.logger.warning('Manally setting particle positions. This is not recommended and could break gradient flow. This also resets particle stress and velocities.')

        pos = pos.clone()

        if len(pos.shape) == 1:
            assert pos.shape == (3,)
            self._tgt['pos'] = self.init_positions_COM_offset + pos

        elif len(pos.shape) == 2:
            assert pos.shape == (self.n, 3)
            self._tgt['pos'] = pos

        else:
            us.raise_exception('Tensor shape not supported.')

    def deactivate(self):
        us.logger.debug(f'{self.__class__.__name__} <{self.id}> deactivated.')
        self._tgt['act'] = us.INACTIVE
        self.active = False

    def activate(self):
        us.logger.debug(f'{self.__class__.__name__} <{self.id}> activated.')
        self._tgt['act'] = us.ACTIVE
        self.active = True

    def _assert_active(self):
        if not self.active:
            us.raise_exception(f'{self.__class__.__name__} is inactive. Call `entity.activate()` first.')

    def _not_created(self, strict=False):
        if strict:
            return self.sim.cur_step_global <= self.step_global_added
        else:
            return self.sim.cur_step_global < self.step_global_added

    def process_input(self):

        # set_pos followed by set_vel, because set_pos resets velocity.
        if self._tgt['pos'] is not None:
            us.assert_contiguous(self._tgt['pos'])
            self._tgt['pos'].assert_sceneless()
            self.set_pos(self.sim.cur_substep_local, self._tgt['pos'])

        if self._tgt['vel'] is not None:
            us.assert_contiguous(self._tgt['vel'])
            self.set_vel(self.sim.cur_substep_local, self._tgt['vel'])

        if self._tgt['act'] is not None:
            assert self._tgt['act'] in [us.ACTIVE, us.INACTIVE]
            self.set_active(self.sim.cur_substep_local, self._tgt['act'])
        
        if self._tgt['col'] is not None:
            us.assert_contiguous(self._tgt['col'])
            self.set_col(self.sim.cur_substep_local, self._tgt['col'])

        for key in self._tgt_keys:
            self._tgt[key] = None

    def sample_cube(self):
        lower = np.array(self.geom.lower)
        filling = self.filling
        target_particle_count = TARGET_PARTICLE_COUNT  # Set this to the expected number of particles

        if self.geom.size is not None:
            upper = lower + np.array(self.geom.size)
        else:
            upper = np.array(self.geom.upper)
        if not (upper >= lower).all():
            us.raise_exception('Invalid lower and upper corner.')

        if filling == 'natural':
            filling = 'grid' # for cube, natural is the same as grid
        positions = self._sample_cube(lower, upper, filling, target_particle_count)

        return positions

    def sample_cylinder(self):
        center  = np.array(self.geom.center)
        radius  = self.geom.radius
        height  = self.geom.height
        filling = self.filling
        target_particle_count = TARGET_PARTICLE_COUNT  # Set this to the expected number of particles

        if filling == 'natural':
            n_y = self.compute_n_particles_1D(height)
            n_r = self.compute_n_particles_1D(radius)
            positions = []
            for y_layer in np.linspace(center[1]-height/2, center[1]+height/2, n_y+1):
                for r_layer in np.linspace(0, radius, n_r+1):
                    n_layer = max(self.compute_n_particles_1D(2*np.pi*r_layer), 1)
                    rad_layer = np.linspace(0, np.pi*2, n_layer+1)[:-1]
                    x_layer = np.cos(rad_layer) * r_layer + center[0]
                    z_layer = np.sin(rad_layer) * r_layer + center[2]
                    positions_layer = np.vstack([x_layer, np.repeat(y_layer, n_layer), z_layer])
                    positions.append(positions_layer)
            positions = np.hstack(positions).T
        else: 
            # sample a cube first
            cube_lower = np.array([center[0] - radius, center[1] - height / 2.0, center[2] - radius])
            cube_upper = np.array([center[0] + radius, center[1] + height / 2.0, center[2] + radius])
            positions = self._sample_cube(cube_lower, cube_upper, filling, target_particle_count)

            # reject out-of-boundary particles
            positions_r = np.linalg.norm(positions[:, [0, 2]] - center[[0, 2]], axis=1)
            positions = positions[positions_r <= radius]

        # If we have more particles than the target, randomly select the target number
        if len(positions) > target_particle_count:
            indices = np.random.choice(len(positions), target_particle_count, replace=False)
            positions = positions[indices]
        elif len(positions) < target_particle_count:
            us.logger.warning(f"Not enough particles generated for cylinder. Expected {target_particle_count}, got {len(positions)}")

        return positions
    
    def sample_supertoroid(self):
        filling = self.filling
        center  = np.array(self.geom.center)
        size    = np.array(self.geom.size)
        hole    = self.geom.hole
        e_lat   = self.geom.e_lat
        e_lon   = self.geom.e_lon
        target_particle_count = TARGET_PARTICLE_COUNT  # Set this to the expected number of particles

        # sample a cube first
        cube_lower = -size/2
        cube_upper = size/2
        positions = self._sample_cube(cube_lower, cube_upper, filling, target_particle_count)

        # reject out-of-boundary particles
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        a1, a2, a3 = size/2
        boundary_condition = (((x*(hole+1)/a1)**(2/e_lon) + (y*(hole+1)/a2)**(2/e_lon))**(e_lon/2) - hole)**(2/e_lat) + (z/a3)**(2/e_lat) - 1
        positions = positions[boundary_condition <= 0]

        positions += center

        # If we have more particles than the target, randomly select the target number
        if len(positions) > target_particle_count:
            indices = np.random.choice(len(positions), target_particle_count, replace=False)
            positions = positions[indices]
        elif len(positions) < target_particle_count:
            us.logger.warning(f"Not enough particles generated for supertoroid. Expected {target_particle_count}, got {len(positions)}")

        return positions

    def sample_mesh(self):
        filling      = self.filling
        file         = self.geom.file
        voxelize_res = self.geom.voxelize_res
        scale        = np.array(self.geom.scale) if hasattr(self.geom.scale, '__iter__') else np.array([self.geom.scale]*3)
        pos          = np.array(self.geom.pos) if hasattr(self.geom.pos, '__iter__') else np.array([0, 0, self.geom.pos])
        target_particle_count = TARGET_PARTICLE_COUNT

        if filling == 'natural':
            filling = 'grid' # for mesh, natural is the same as grid

        us.logger.debug(f"Attempting to get voxelized mesh path for file: {file}")
        voxelized_file_path = get_voxelized_mesh_path(file, voxelize_res)
        if voxelized_file_path is None:
            us.logger.error(f"Failed to get voxelized mesh path for file: {file}")
            us.raise_exception(f"Failed to get voxelized mesh path for file: {file}")

        us.logger.debug(f"Voxelized mesh path obtained: {voxelized_file_path}")

        # Load the voxelized mesh data
        with open(voxelized_file_path, 'rb') as f:
            voxel_matrix = pickle.load(f)

        voxel_shape = np.array(voxel_matrix.shape)

        # Adjust scale to fit within -0.4 to 0.4 for x and y
        xy_scale = 0.8 / max(voxel_shape[0], voxel_shape[1])
        scale = np.array([xy_scale, xy_scale, scale[2]])

        # Adjust position
        epsilon = 1e-4
        pos[2] = max(pos[2], scale[2] * voxel_shape[2] / 2) + epsilon
        pos[0] = pos[1] = 0

        cube_lower = np.array([-0.4, -0.4, pos[2] - scale[2] * voxel_shape[2] / 2])
        cube_upper = np.array([0.4, 0.4, pos[2] + scale[2] * voxel_shape[2] / 2])
        
        oversampling_factor = 2
        initial_particle_count = target_particle_count * oversampling_factor
        positions = self._sample_cube(cube_lower, cube_upper, filling, initial_particle_count)

        # Convert positions to voxel coordinates
        voxel_coords = ((positions - cube_lower) / (cube_upper - cube_lower) * voxel_shape).astype(int)
        voxel_coords = np.clip(voxel_coords, 0, voxel_shape - 1)
        
        # Check if any coordinates are out of bounds
        valid_coords = np.all((voxel_coords >= 0) & (voxel_coords < voxel_shape), axis=1)
        
        # Only keep valid coordinates
        voxel_coords = voxel_coords[valid_coords]
        positions = positions[valid_coords]

        # Use valid coordinates to index into voxel matrix
        mask = voxel_matrix[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        positions = positions[mask]

        # Final check to ensure all particles are within bounds
        solver_lower = np.array([-0.4, -0.4, epsilon])
        solver_upper = np.array([0.4, 0.4, 1.0])
        within_bounds = np.all((positions >= solver_lower) & (positions <= solver_upper), axis=1)
        positions = positions[within_bounds]

        # Adjust particle count to match TARGET_PARTICLE_COUNT
        if len(positions) > TARGET_PARTICLE_COUNT:
            indices = np.random.choice(len(positions), TARGET_PARTICLE_COUNT, replace=False)
            positions = positions[indices]
        elif len(positions) < TARGET_PARTICLE_COUNT:
            us.logger.warning(f"Not enough particles generated. Expected {TARGET_PARTICLE_COUNT}, got {len(positions)}")
            # If we don't have enough particles, we'll duplicate existing ones
            additional_particles_needed = TARGET_PARTICLE_COUNT - len(positions)
            additional_indices = np.random.choice(len(positions), additional_particles_needed, replace=True)
            additional_positions = positions[additional_indices]
            # Add small random offsets to avoid exact duplicates
            additional_positions += np.random.uniform(-self.particle_diameter/100, self.particle_diameter/100, additional_positions.shape)
            positions = np.vstack([positions, additional_positions])

        us.logger.debug(f"Mesh sampling completed. Number of particles: {len(positions)}")
        return positions
    
    def sample_particles(self):
        positions = np.asarray(self.geom.centers).reshape(-1, 3)
        # jitter (as in generation)
        max_offset = self.particle_diameter/2  # in any direction
        positions += np.random.uniform(low=-max_offset, high=max_offset, size=positions.shape)
        # clip z to be above the floor
        positions = positions[positions[:, 2] >= 0.0]
        return positions

    @property
    def n(self):
        return len(self.init_positions)

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'id : {_repr_uuid(self.id)}\n' \
               f'n  : {_repr(self.n)}'
