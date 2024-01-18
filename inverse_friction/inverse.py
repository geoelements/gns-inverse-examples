import sys
import time
import json
import os
import argparse
import numpy as np
from utils import To_Torch_Model_Param
import torch.utils.checkpoint
from forward import rollout_with_checkpointing
from utils import make_animation
from utils import compute_penalty
from utils import visualize_final_deposits
from run_mpm import run_mpm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import reading_utils
from gns import data_loader
from gns import train
from inverse_friction.convert_hd5_to_npz import convert_hd5_to_npz


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="inverse_friction/short_phi21/config.json", type=str, help="Path to input json file (e.g., `data/config.json`")
args = parser.parse_args()

# Read config file
with open(args.input_path, 'r') as file:
    config = json.load(file)

simulation_name = config["simulation_name"]
path = f"inverse_friction/data/{simulation_name}/"
ground_truth_npz = config["ground_truth_npz"]

# Iterations
niteration = config["niteration"]
resume = config["resume"]["enabled"]
resume_iteration = config["resume"]["resume_iteration"]

# Diff method (`from_mpm` or `from_same_5vels`)
if config["diff_method"]["fd"] == True and config["diff_method"]["ad"] == False:
    diff_method = "fd"  # ad or fd
    dphi = config["diff_method"]["fd_config"]["dphi"]
elif config["diff_method"]["ad"] == True and config["diff_method"]["fd"] == False:
    diff_method = "ad"
else:
    raise ValueError("diff method should be either `fd` or `ad`")

# inputs for MPM to make X0 (i.e., p0, p1, p2, p3, p4, p5)
if config["x0_mode"]["from_same_5vels"] == True and config["x0_mode"]["from_mpm"] == False:
    x0_mode = "from_same_5vels"
elif config["x0_mode"]["from_mpm"] == True and config["x0_mode"]["from_same_5vels"] == False:
    x0_mode = "from_mpm"
    mpm_config = config["x0_mode"]["from_mpm"]["mpm_config"]
    uuid_name = mpm_config["uuid_name"]
    mpm_input = mpm_config["mpm_input"]  # mpm input file to start running MPM for phi & phi+dphi
    analysis_dt = mpm_config["analysis_dt"]
    output_steps = mpm_config["output_steps"]
    analysis_nsteps = output_steps * 5 + 1  # only run to get 6 initial positions to make X_0 in GNS
    ndim = mpm_config["ndim"]
else:
    raise ValueError("diff method should be either `from_same_5vels` or `from_mpm`")

# Optimizer
inverse_timestep = config["inverse_timestep"]
lr = config["lr"]  # learning rate (phi=21: 500, phi=42: 1000)
phi = config["phi"]  # initial guess of phi, default=30.0
if config["loss_constraint"]["regularization"] == True:
    loss_regularization = True
    loss_limit = config["loss_constraint"]['loss_limit']  # default=0.0005
    penalty_mag = config["loss_constraint"]['penalty_mag']  # default=4000
else:
    loss_regularization = False
    loss_limit = config["loss_constraint"]['loss_limit']  # used for stop further iteration
    penalty_mag = "none"  # default=4000
if config["noise_data"]["enabled"] == True:
    noise_data = True
    runout_mean = 0.5
    runout_std = 0.01 * runout_mean
else:
    noise_data = False
    runout_mean = "none"
    runout_std = "none"

# Forward simulator
checkpoint_interval = config["simulator"]["checkpoint_interval"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_std = config["simulator"]["noise_std"] # hyperparameter used to train GNS.
NUM_PARTICLE_TYPES = 9
model_path = config["simulator"]["model_path"]
simulator_metadata_path = config["simulator"]["simulator_metadata_path"]
model_file = config["simulator"]["model_file"]

# outputs
if noise_data:
    output_dir = f"/outputs_{diff_method}_const_lim{loss_limit}_mag{penalty_mag}_lr{lr}_noised/"
else:
    output_dir = f"/outputs_{diff_method}_const_lim{loss_limit}_mag{penalty_mag}_lr{lr}/"
save_step = config["outputs"]["save_step"]


# ---------------------------------------------------------------------------------

# Load simulator
metadata = reading_utils.read_metadata(simulator_metadata_path, "rollout")
simulator = train._get_simulator(metadata, noise_std, noise_std, device)
if os.path.exists(model_path + model_file):
    simulator.load(model_path + model_file)
else:
    raise Exception(f"Model does not exist at {model_path + model_file}")
simulator.to(device)
simulator.eval()

# Get ground truth particle position at the inversion timestep
mpm_trajectory = [item for _, item in np.load(f"{path}/{ground_truth_npz}", allow_pickle=True).items()]
target_positions = torch.tensor(mpm_trajectory[0][0])
target_final_runout = target_positions[inverse_timestep][:, 0].max().to(device)

# Initialize friction angle to start optimizing
friction = torch.tensor([phi], requires_grad=True, device=device)
friction_model = To_Torch_Model_Param(friction)

# Set up the optimizer
optimizer = torch.optim.SGD(friction_model.parameters(), lr=lr)

# Set output folder
if not os.path.exists(f"{path}/{output_dir}"):
    os.makedirs(f"{path}/{output_dir}")

# Save input config to the current output dir
with open(f'{path}/{output_dir}/config.json', 'w') as file:
    json.dump(config, file, indent=4)

# Resume
if resume:
    print(f"Resume from the previous state: iteration{resume_iteration}")
    checkpoint = torch.load(f"{path}/{output_dir}/optimizer_state-{resume_iteration}.pt")
    start_iteration = checkpoint["iteration"]
    friction_model.load_state_dict(checkpoint['friction_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    start_iteration = 0
friction = friction_model.current_params

# Start optimization iteration
for iteration in range(start_iteration+1, niteration):
    start = time.time()
    optimizer.zero_grad()  # Clear previous gradients

    if x0_mode == "from_mpm":
        # Run MPM with current friction angle to get X0
        run_mpm(path,
                output_dir,
                mpm_input,
                iteration,
                friction.item(),
                analysis_dt,
                analysis_nsteps,
                output_steps)

        # Make `.npz` to prepare initial state X_1 for rollout
        convert_hd5_to_npz(path=f"{path}/{output_dir}/mpm_iteration-{iteration}/",
                           uuid=f"/results/{uuid_name}/",
                           ndim=ndim,
                           output=f"{path}/{output_dir}/x0_iteration-{iteration}.npz",
                           material_feature=True,
                           dt=1.0)

    # Load data containing X0, and get necessary features.
    # First, obtain ground truth features except for material property
    if x0_mode == "from_mpm":
        dinit = data_loader.TrajectoriesDataset(path=f"{path}/{output_dir}/x0_iteration-{iteration}.npz")
    elif x0_mode == "from_same_5vels":
        dinit = data_loader.TrajectoriesDataset(path=f"{path}/{ground_truth_npz}")
    else:
        raise ValueError("x0_mode should be either `from_mpm` or `from_same_5vels`")

    # Get initial features for GNS
    for example_i, features in enumerate(dinit):  # only one item exists in `dint`. No need `for` loop
        if len(features) < 3:
            raise NotImplementedError("Data should include material feature")
        initial_positions = features[0][:, :6, :].to(device)
        particle_type = features[1].to(device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)

    friction_before_update = friction.item()
    # Do forward pass as compute gradient of parameter
    if diff_method == "ad":
        # Make material property feature from current phi
        material_property_tensor = torch.tan((friction * torch.pi / 180)) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()

        print("Start rollout...")
        start_time_forward = time.time()
        predicted_positions = rollout_with_checkpointing(
            simulator=simulator,
            initial_positions=initial_positions,
            particle_types=particle_type,
            material_property=material_property_tensor,
            n_particles_per_example=n_particles_per_example,
            nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
            checkpoint_interval=checkpoint_interval,
        )
        end_time_forward = time.time()
        time_ad_forward = end_time_forward - start_time_forward
        print(f"Forward rollout took {time_ad_forward}s")

        if noise_data == False:
            inversion_runout = predicted_positions[inverse_timestep, :, 0].max()
        elif noise_data == True:
            inversion_runout = predicted_positions[inverse_timestep, :, 0].max() \
                               + torch.randn(1).to(device) * runout_std

        loss = (inversion_runout - target_final_runout) ** 2
        if loss_regularization:
            penalty = compute_penalty(loss, threshold=loss_limit, alpha=penalty_mag)
            loss = loss + penalty

        time_start_backprop = time.time()
        print("Backpropagate...")
        loss.backward()
        time_end_backprop = time.time()
        time_ad_backprop = time_end_backprop - time_start_backprop
        print(f"Backpropagate took {time_ad_backprop}s")

    elif diff_method == "fd":  # finite diff
        # Prepare (phi, phi+dphi)
        material_property_tensor = torch.tan((friction * torch.pi / 180)) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()
        material_property_tensor_perturb = torch.tan((friction+dphi) * torch.pi / 180) * torch.full(
            (len(initial_positions), 1), 1, device=device).to(torch.float32).contiguous()

        # Rollout at [phi & phi+dphi]
        with torch.no_grad():
            print("Start rollout with phi")
            t0 = time.time()
            predicted_positions = rollout_with_checkpointing(
                simulator=simulator,
                initial_positions=initial_positions,
                particle_types=particle_type,
                material_property=material_property_tensor,
                n_particles_per_example=n_particles_per_example,
                nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
                checkpoint_interval=checkpoint_interval,
                is_checkpointing=False
            )
            t1 = time.time()
            time_fd_phi0 = t1 - t0

            print("Start rollout with phi+dphi")
            t0 = time.time()
            predicted_positions_perturb = rollout_with_checkpointing(
                simulator=simulator,
                initial_positions=initial_positions,
                particle_types=particle_type,
                material_property=material_property_tensor_perturb,
                n_particles_per_example=n_particles_per_example,
                nsteps=inverse_timestep - initial_positions.shape[1] + 1,  # exclude initial positions (x0) which we already have
                checkpoint_interval=checkpoint_interval,
                is_checkpointing=False
            )
            t1 = time.time()
            time_fd_phi_1 = t1 - t0

        # Compute gradient of loss: (loss(phi+dphi) - loss(phi))/dphi
        if noise_data == False:
            inversion_runout = predicted_positions[inverse_timestep, :, 0].max()
            inversion_runout_perturb = predicted_positions_perturb[inverse_timestep, :, 0].max()
        if noise_data == True:
            inversion_runout = predicted_positions[inverse_timestep, :, 0].max() \
                               + torch.randn(1).to(device) * runout_std
            inversion_runout_perturb = predicted_positions_perturb[inverse_timestep, :, 0].max() \
                                       + torch.randn(1).to(device) * runout_std

        loss = (inversion_runout - target_final_runout) ** 2
        loss_perturb = (inversion_runout_perturb - target_final_runout) ** 2
        gradient = (loss_perturb - loss) / dphi
        friction.grad = torch.tensor([gradient], dtype=friction.dtype, device=friction.device)

    else:
        raise NotImplementedError

    # Visualize current prediction
    print(f"iteration {iteration-1}, Friction {friction.item():.5f}, Loss {loss.item():.8f}")
    visualize_final_deposits(predicted_positions,
                             target_positions,
                             metadata,
                             write_path=f"{path}/{output_dir}/inversion-{iteration-1}.png",
                             friction=friction.item())

    # Perform optimization step
    optimizer.step()

    end = time.time()
    time_for_iteration = end - start

    # Save and report optimization status
    if iteration % save_step == 0:

        # Make animation at the last iteration
        if iteration == niteration-1:
            print(f"Rendering animation at {iteration}...")
            positions_np = np.concatenate(
                (initial_positions.permute(1, 0, 2).detach().cpu().numpy(),
                 predicted_positions.detach().cpu().numpy())
            )
            make_animation(positions=positions_np,
                           boundaries=metadata["bounds"],
                           output=f"{path}/{output_dir}/animation-{iteration}.gif",
                           timestep_stride=5)

        # Save optimizer state
        if diff_method == "ad":
            time_fd_phi0, time_fd_phi_1 = 'none', 'none'
        if diff_method == 'fd':
            time_ad_forward, time_ad_backprop = 'none', 'none'
        torch.save({
            'iteration': iteration,
            'time_spent': {
                "time_for_iteration": time_for_iteration,
                "time_ad": {"forward": time_ad_forward, "backprop": time_ad_backprop},
                "time_fd": {"forward1": time_fd_phi0, "forward2": time_fd_phi_1}
            },
            'position_state_dict': {
                "target_positions": target_positions.clone().detach().cpu().numpy(),
                "inversion_positions": predicted_positions.clone().detach().cpu().numpy()
            },
            'friction_before_update': friction_before_update,
            'friction_state_dict': To_Torch_Model_Param(friction).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'grad': friction.grad.item(),
            'loss_regularization': {"loss_limit": loss_limit, "penalty_mag": penalty_mag} if loss_regularization else None,
            'dphi': dphi if diff_method == "fd" else None
        }, f"{path}/{output_dir}/optimizer_state-{iteration}.pt")

    if loss < loss_limit:
        print(f"Loss reached lower than {loss_limit}")
        break