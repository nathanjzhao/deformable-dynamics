Dataset structure:
- The top-level groups are the dataset splits (`train`, `val_test`, `real`).
- The groups in each split correspond to the keys of a sample; stored as arrays of shape `{num_scenes} x {num_frames} x ...`.

Dataset usage example:
- Assuming `hdf5plugin` and `hdf5` are installed in the python environment, loading the `real` split is done by:

```
import hdf5plugin
import h5py

subset = 'real'  # one of 'train', 'val_test', 'real'
scene_step = 1  # optionally subsample scenes
frame_step = 1  # optionally subsample frames

with h5py.File('/path/to/doughnet/dataset.h5', 'r', libver='latest', swmr=True) as f:
    dataset = {k: f[subset][k][::scene_step, ::frame_step] for k in f[subset].keys()}

print(f'Loaded the {subset} split with {dataset["scene"].shape[0]} scenes of {dataset["scene"].shape[1]} frames each.')
```
