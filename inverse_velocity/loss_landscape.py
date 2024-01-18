import time
import sys
import os
import numpy as np
import json
import pickle
import torch.utils.checkpoint

sys.path.append("/work2/08264/baagee/frontera/gns-main/")
from example.inverse_velocity.forward import rollout_with_checkpointing

from gns import reading_utils
from gns import data_loader
from gns import train


# Work dir
path = "data/"

# Forward Simulator input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = "data/"
model_file = "model-7020000.pt"
simulator_metadata_path = "data/"
simulator_metadata_file = "gns_metadata.json"
checkpoint_interval = 500

# loss constraint
regularization = False
loss_limit = 0.0005  # default=0.0005
penalty_mag = 4000  # default=4000

# Ground Truth input
ground_truth_npz = "sand2d_inverse_eval15.npz"

# Output dir
output_dir = "data/outputs/"

# Eval phi points
# frictions = torch.linspace(17, 47, 100).to(device)
frictions = torch.linspace(17, 47, 100).to(device)
inverse_timestep = 379

# Data container
save_data = {"friction": [], "loss": []}


# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path, "rollout", file_name=simulator_metadata_file)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth particle position at the inversion timestep
mpm_trajectory = [item for _, item in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).items()]
target_final_positions = torch.tensor(
    mpm_trajectory[0][0][-1], device=device)
target_final_runout = target_final_positions[:, 0].max()

# Get initial condition
dinit = data_loader.TrajectoriesDataset(path=f"{path}/{ground_truth_npz}")
# Get initial features for GNS
for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need `for` loop
    if len(features) < 3:
        raise NotImplementedError("Data should include material feature")
    initial_positions = features[0][:, :6, :].to(device)
    particle_type = features[1].to(device)
    n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)

for friction in frictions:
    # Make material property feature from current phi
    material_property_tensor = torch.tan((friction * torch.pi / 180)) * torch.full(
        (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()

    with torch.no_grad():
        predicted_positions = rollout_with_checkpointing(
            simulator=simulator,
            initial_positions=initial_positions,
            particle_types=particle_type,
            material_property=material_property_tensor,
            n_particles_per_example=n_particles_per_example,
            nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
            checkpoint_interval=checkpoint_interval
        )
    pred_final_runout = predicted_positions[inverse_timestep, :, 0].max()

    loss = (pred_final_runout - target_final_runout) ** 2
    if regularization:
        penalty = torch.pow(torch.relu(loss_limit - loss), 2) * penalty_mag
        loss = loss + penalty

    save_data["loss"].append(loss.item())
    save_data["friction"].append(friction.item())

# Save data
fileObj = open('../../../paper/gns-inverse/data/loss-friction-hist-tall_phi42.pkl', 'wb')
pickle.dump(save_data, fileObj)
fileObj.close()


