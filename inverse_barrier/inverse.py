import argparse
import os
import json
import sys
import optimizations
from utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inverse_velocity.utils import To_Torch_Model_Param
from gns import reading_utils
from gns import train


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="inverse_barrier/config.json", type=str,
                    help="Path to input json file (e.g., `data/config.json`")
args = parser.parse_args()


# Read the JSON configuration file
with open(args.input_path, 'r') as file:
    config = json.load(file)

path = config["path"]

# Inputs for optimization
optimizer_type = config["optimizer"]["type"]  # adam or lbfgs
loss_mesure = config["optimizer"]["loss_measure"]  # `farthest_positions` or `centroid`
niterations = config["optimizer"]["niterations"]
lr = config["optimizer"]["lr"]  # use 0.5 - 1.0 for lbfgs and 0.01 or smaller to adam
# initial location guess of barriers
barrier_locations = config["initial_guess"]["barrier_locations"]  # x and z at lower edge
# prescribed constraints for barrier geometry
barrier_info = config["barrier_info"]
n_farthest_particles = config["n_farthest_particles"]

# inputs for ground truth
ground_truth_npz = config["ground_truth"]["npz_file"]
ground_truth_mpm_inputfile = config["ground_truth"]["mpm_inputfile"]

# Inputs for forward simulator
nsteps = config["simulator"]["nsteps"]
checkpoint_interval = config["simulator"]["checkpoint_interval"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = 6.7e-4  # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
dt_mpm = config["simulator"]["dt_mpm"]
model_path = config["simulator"]["model_path"]
model_file = config["simulator"]["model_file"]
simulator_metadata_path = config["simulator"]["metadata_path"]
simulator_metadata_file = config["simulator"]["metadata_file"]

# Resume options
resume = config["resume"]["enabled"]
resume_epoch = config["resume"]["epoch"]

# Save options
output_dir = config["save_options"]["output_dir"]
save_step = config["save_options"]["save_step"]

# Set output folder
if not os.path.exists(f"{output_dir}"):
    os.makedirs(f"{output_dir}")
# Save input config to the current output dir
with open(f'{output_dir}/config.json', 'w') as file:
    json.dump(config, file, indent=4)

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path, "rollout", file_name=simulator_metadata_file)
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth barrier positions
f = open(f"{path}/{ground_truth_mpm_inputfile}")
mpm_inputs = json.load(f)
barrier_geometries_true = mpm_inputs["gen_cube_from_data"]["sim_inputs"][0]["obstacles"]["cubes"]
barrier_locs_true = []
for geometry in barrier_geometries_true:
    loc = [geometry[0], geometry[2]]
    barrier_locs_true.append(loc)

# Get particle positions
mpm_trajectory = [item for _, item in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).items()]
positions = torch.tensor(mpm_trajectory[0][0])
particle_types = torch.tensor(mpm_trajectory[0][1])

# Get positions by particle type
kinematic_positions, stationary_positions = get_positions_by_type(
    positions, particle_types)

# Generate barrier mass filled with material points
dist_between_particles = mpm_inputs["domain_size"]/mpm_inputs["sim_resolution"][0]/2
barrier_particles = fill_cuboid_with_particles(
    len_x=barrier_info["barrier_width"],
    len_y=barrier_info["barrier_height"],
    len_z=barrier_info["barrier_width"],
    spacing=dist_between_particles).to(device)

# Get ground truth particle positions at the last timestep for `n_farthest_particles` to compute loss
runout_end_true = get_runout_end(
    kinematic_positions[-1], n_farthest_particles).to(device)
# Get ground truth centroid of particle positions at the last timestep
# centroid_true = torch.mean(kinematic_positions[-1], dim=0).to(device)
centroid_true = torch.mean(runout_end_true, dim=0).to(device)

# Initialize barrier locations
barrier_zlocs_torch = torch.tensor(
    [barrier_locations[0][1], barrier_locations[1][1]],
    requires_grad=True, device=device)
barrier_zlocs_param = To_Torch_Model_Param(barrier_zlocs_torch)

# Set up the optimizer
if optimizer_type == "lbfgs":
    optimizer = torch.optim.LBFGS(barrier_zlocs_param.parameters(),
                                  lr=lr, max_iter=niterations, history_size=100)
elif optimizer_type == "adam":
    optimizer = torch.optim.Adam(barrier_zlocs_param.parameters(), lr=lr)
else:
    raise ValueError("Check `optimizer_type`")

# Resume TODO (yc): depends on optimizer type
if resume:
    print(f"Resume from the previous state: iteration{resume_epoch}")
    checkpoint = torch.load(f"{output_dir}/optimizer_state-e{resume_epoch}.pt")
    start_epoch = checkpoint["iteration"]
    barrier_zlocs_param.load_state_dict(checkpoint['updated_barrier_loc_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    start_epoch = 0
barrier_zlocs = barrier_zlocs_param.current_params

# Start optimization iteration
if optimizer_type == "lbfgs":
    optimizations.lbfgs(
        loss_mesure,
        simulator,
        nsteps,
        mpm_inputs,
        niterations,
        checkpoint_interval,
        runout_end_true,
        centroid_true,
        n_farthest_particles,
        kinematic_positions,
        barrier_info,
        barrier_particles,
        barrier_zlocs,
        barrier_locs_true,  # list
        optimizer,
        output_dir,
        device)

elif optimizer_type == "adam":
    for iteration in range(start_epoch, niterations):
        barrier_zlocs, _ = optimizations.adam(
            loss_mesure,
            simulator,
            nsteps,
            mpm_inputs,
            iteration,
            checkpoint_interval,
            runout_end_true,
            centroid_true,
            n_farthest_particles,
            kinematic_positions,
            barrier_info,
            barrier_particles,
            barrier_zlocs,
            barrier_locs_true,
            optimizer,
            output_dir,
            device)

else:
    raise ValueError("Check `optimizer type`")


    # TODO (yc): When to save animation?
    # # Save animation after iteration
    # if iteration % save_step == 0:
    #     render_animation(
    #         predicted_positions,
    #         current_particle_type,
    #         mpm_inputs,
    #         timestep_stride=10,
    #         write_path=f"{output_dir}/trj-{iteration}.gif")






