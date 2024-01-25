import numpy as np
import alphashape
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import torch
from shapely.geometry import Polygon, MultiPolygon


KINEMATIC_PARTICLE = 6
STATIONARY_PARTICLE = 3


def render_animation(
        positions: torch.tensor,
        particle_type,
        mpm_inputs,
        write_path,
        timestep_stride=5,
):

    # Start with computing displacement
    kinematic_positions, stationary_positions = get_positions_by_type(
        positions,
        particle_type)

    kinematic_positions = kinematic_positions.detach().cpu().numpy()
    stationary_positions = stationary_positions.detach().cpu().numpy()

    initial_kinematic_position = kinematic_positions[None, 0]
    initial_kinematic_positions = np.repeat(initial_kinematic_position, repeats=len(positions), axis=0)
    disp = np.linalg.norm(kinematic_positions - initial_kinematic_positions, axis=-1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def animate(i):
        print(f"Render step {i}/{len(positions)}")
        fig.clear()

        cmap = plt.cm.viridis
        vmax = np.ndarray.flatten(disp).max()
        vmin = np.ndarray.flatten(disp).min()
        sampled_value = disp[i]

        # Note: z and y is interchanged to match taichi coordinate convention.
        ax = fig.add_subplot(projection='3d', autoscale_on=False)
        ax.set_xlim(mpm_inputs['sim_space'][0])
        ax.set_ylim(mpm_inputs['sim_space'][2])
        ax.set_zlim(mpm_inputs['sim_space'][1])
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.invert_zaxis()

        trj = ax.scatter(kinematic_positions[i][:, 0],
                         kinematic_positions[i][:, 2],
                         kinematic_positions[i][:, 1],
                         c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
        ax.scatter(stationary_positions[i][:, 0],
                   stationary_positions[i][:, 2],
                   stationary_positions[i][:, 1], c="black")
        fig.colorbar(trj)

        ax.set_box_aspect(
            aspect=(float(mpm_inputs['sim_space'][0][0]) - float(mpm_inputs['sim_space'][0][1]),
                    float(mpm_inputs['sim_space'][2][0]) - float(mpm_inputs['sim_space'][2][1]),
                    float(mpm_inputs['sim_space'][1][0]) - float(mpm_inputs['sim_space'][1][1])))
        ax.view_init(elev=20., azim=i*0.5)
        # ax.view_init(elev=20., azim=0.5)
        ax.grid(True, which='both')

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, len(positions), timestep_stride), interval=20)

    ani.save(write_path, dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {write_path}.gif")


def visualize_state(
        vis_data: dict,
        barrier_info: dict,
        mpm_inputs: dict,
        loss: float,
        write_path: str
):
    """
    Make runout comparison plot
    Args:
        vis_data (dict): data for visualization following a specific format
        barrier_info (dict): prescribed barrier geometry information.
            It is used to make a rectangular patch in fig to represent barriers
        mpm_inputs (dict): used to get simulation domain boundary
        loss (float):
        write_path (str):

    Returns:

    """

    barrier_height = barrier_info["barrier_height"]
    barrier_width = barrier_info["barrier_width"]

    vis_params = {
        "pred": {
            "label": "Pred",
            "perimeter_color": "black",
            "particle_color": "purple",
            "alpha": 0.5
        },
        "true": {
            "label": "True",
            "perimeter_color": "darkorange",
            "particle_color": "yellow",
            "alpha": 0.5
        }
    }

    # Init fig
    fig, ax = plt.subplots()

    for i, (key, value) in enumerate(vis_data.items()):
        # Preprocess data
        last_position = value["kinematic_positions"][-1].detach().cpu().numpy()
        runout_perimeter = alphashape.alphashape(
            last_position[:, [0, 2]], alpha=20.0)  # smaller alpha fit more tight
        if "runout_end" in value:
            runout_end = value["runout_end"].detach().cpu().numpy()
        if "centroid" in value:
            centroid = value["centroid"].detach().cpu().numpy()

        # Get rectangular patches for pred barriers
        barrier_patches = []
        barrier_locs = value["barrier_locs"].detach().cpu().numpy()
        for barrier_edge in barrier_locs:
            barrier_patch = Rectangle(
                xy=barrier_edge,
                width=barrier_width,
                height=barrier_width,
                edgecolor=vis_params[key]["perimeter_color"],
                fill=False,
                linewidth=1.5,
                zorder=10
            )
            barrier_patches.append(barrier_patch)

        # Plot runout
        ax.scatter(last_position[:, 0], last_position[:, 2],
                   alpha=0.3, s=2.0, c=vis_params[key]["particle_color"])

        # Plot runout end used in loss calculation
        if "runout_end" in value:
            ax.scatter(runout_end[:, 0], runout_end[:, 2],
                       s=5.0,
                       c=vis_params[key]["particle_color"],
                       edgecolors=vis_params[key]["perimeter_color"])
        if "centroid" in value:
            ax.scatter(centroid[0], centroid[2],
                       s=50.0,
                       c=vis_params[key]["particle_color"],
                       edgecolors=vis_params[key]["perimeter_color"],
                       zorder=11)

        ### Plot runout fill. Simple `PolygonPath` doesn't work in the current version of the package
        # ax.add_patch(PolygonPatch(runout_perimeter, alpha=0.2))
        if isinstance(runout_perimeter, Polygon):
            x, y = runout_perimeter.exterior.xy
            ax.fill(x, y, alpha=0.2,
                    color=vis_params[key]["perimeter_color"],
                    label=vis_params[key]["label"])
        elif isinstance(runout_perimeter, MultiPolygon):
            for geom in runout_perimeter.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, alpha=0.2,
                        color=vis_params[key]["perimeter_color"],
                        label=vis_params[key]["label"])
        ###

        for barrier_patch in barrier_patches:
            ax.add_patch(barrier_patch)

    ax.set_xlim(mpm_inputs['sim_space'][0])
    ax.set_ylim(mpm_inputs['sim_space'][2])
    ax.set_aspect("equal")
    ax.set_title(f"Loss: {loss:.3e}")
    ax.legend()
    ax.grid(True)
    # plt.show()
    plt.savefig(write_path)


def get_features(
        kinematic_positions: torch.tensor,
        barrier_particles: torch.tensor,
        device: torch.device
):
    """
    Get features for gns rollout
    Args:
        kinematic_positions (torch.tensor): shape=(timesteps, nparticles, dims)
        barrier_particles (torch.tensor): shape=(nparticles, dims)
        device (torch.device):

    Returns:
        initial_positions (torch.tensor): shape=(nparticles, timesteps, dims)
        particle_type (torch.tensor): shape=(nparticles, )
        n_particles_per_example (torch.tensor)
    """
    # Get initial positions
    initial_kinematic_positions = kinematic_positions[:6].to(device)
    barrier_particles_unsqueeze = barrier_particles.unsqueeze(0)
    current_stationary_positions = barrier_particles_unsqueeze.repeat(6, 1, 1)
    # Note that the shape is changed to (nparticles, timesteps, dims) from (timesteps, nparticles, dims)
    #   to follow GNS rollout function convention
    initial_positions = torch.concat(
        (initial_kinematic_positions, current_stationary_positions), dim=1
    ).permute(1, 0, 2).to(torch.float32).contiguous()

    # Make particle type
    kinematic_particle_type = torch.full(
        (initial_kinematic_positions.shape[1], ), KINEMATIC_PARTICLE)
    stationary_particle_type = torch.full(
        (current_stationary_positions.shape[1], ), STATIONARY_PARTICLE)
    particle_type = torch.cat(
        (kinematic_particle_type, stationary_particle_type)).contiguous().to(device)

    # Get number of particles
    n_particles_per_example = torch.tensor([len(particle_type)], dtype=torch.int32).to(device)

    return initial_positions, particle_type, n_particles_per_example


def locate_barrier_particles(
        barrier_particles: torch.tensor,
        barrier_locations: torch.tensor,
        base_height: torch.tensor
):
    # Make current barrier particles with the current locations
    current_barrier_groups = []
    for loc in barrier_locations:
        moved_particles = barrier_particles.clone()
        # move `barrier_particles` corresponding to the current location
        moved_particles[:, 0] += loc[0]
        moved_particles[:, 1] += base_height
        moved_particles[:, 2] += loc[1]
        current_barrier_groups.append(moved_particles)
    current_barrier_particles = torch.vstack(current_barrier_groups)

    return current_barrier_particles


def get_runout_end(last_kinematic_positions, n_farthest_particles):
    """
    Get particle positions at the last timestep for `n_farthest_particles` to compute loss
    Args:
        last_kinematic_positions (torch.tensor): particle positions at the last timestep of flow
        n_farthest_particles (int): n particles to sample from the farthest x positions

    Returns:
    `n_farthest_particles` sampled from the last particle positions
    """

    # sort particle positions at the last timestep in ascending order
    kinematic_particles_idx_sorted = torch.argsort(last_kinematic_positions[:, 0])
    # get idx of the last `n_farthest_particles`
    runout_end_idx = kinematic_particles_idx_sorted[-n_farthest_particles:]
    # get `runout_end` consisting of `n_farthest_particles` of the runout
    runout_end = last_kinematic_positions[runout_end_idx, :]

    return runout_end


def get_positions_by_type(
        positions: torch.tensor,
        particle_types: torch.tensor):

    kinematic_particles_idx = torch.where(particle_types == KINEMATIC_PARTICLE)[0]
    stationary_particles_idx = torch.where(particle_types == STATIONARY_PARTICLE)[0]
    kinematic_positions = positions[:, kinematic_particles_idx, :]
    stationary_positions = positions[:, stationary_particles_idx, :]

    return kinematic_positions, stationary_positions


def fill_cuboid_with_particles(
        len_x,
        len_y,
        len_z,
        spacing,
        random_disturbance_mag=0.8,
):
    """
    Fill a cuboid with particles using PyTorch tensors and linspace to avoid graph breakage.

    Parameters:
    - len_x: Tensor scalar representing the length of the cuboid along the x-axis.
    - len_y: Tensor scalar representing the height of the cuboid along the y-axis.
    - len_z: Tensor scalar representing the width of the cuboid along the z-axis.
    - spacing: Tensor scalar representing the distance between adjacent particles.
    - random_disturbance_mag: Tensor scalar for randomization magnitude to disturb particle distribution

    Returns:
    - A PyTorch tensor, where each row represents the (x, y, z) coordinates of a particle.
    """

    # Compute nparticles per dim
    n_particles_x = int(len_x / spacing)
    n_particles_y = int(len_y / spacing)
    n_particles_z = int(len_z / spacing)

    # Generate grid of points
    # Assuming particles are located as follows: | -- * ---- * -- |,
    #   cell size is cellsize=domain/resolution (e.g., 2.0/64),
    #   particle spacing is spacing=2.0/64/2,
    #   starting point of particle location is spacing/2
    x = np.linspace(spacing/2, len_x - spacing/2, n_particles_x)
    y = np.linspace(spacing/2, len_y - spacing/2, n_particles_y)
    z = np.linspace(spacing/2, len_z - spacing/2, n_particles_z)

    # Generate meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten and stack to create a list of coordinates
    particles = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    # Disturbance
    disturbance = np.random.uniform(-spacing/2 * random_disturbance_mag,
                                    spacing/2 * random_disturbance_mag,
                                    size=particles.shape)
    disturbed_particles = particles + disturbance
    torch_particles = torch.tensor(disturbed_particles, requires_grad=True)

    return torch_particles


def visualize_particles(particles_array):
    """
    Visualize the particles within a cuboid using a 3D scatter plot.

    Parameters:
    - particles_array: A NumPy array with shape (N, 3) representing the particles' coordinates.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of particles
    ax.scatter(particles_array[:, 0], particles_array[:, 1], particles_array[:, 2])

    # Labeling the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set equal aspect ratio for all axes
    max_range = np.array([particles_array[:, 0].max() - particles_array[:, 0].min(),
                          particles_array[:, 1].max() - particles_array[:, 1].min(),
                          particles_array[:, 2].max() - particles_array[:, 2].min()]).max() / 2.0

    mid_x = (particles_array[:, 0].max() + particles_array[:, 0].min()) * 0.5
    mid_y = (particles_array[:, 1].max() + particles_array[:, 1].min()) * 0.5
    mid_z = (particles_array[:, 2].max() + particles_array[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

# # Example usage
# # Example usage
# len_x = 0.1 # Length of the cuboid along the x-axis
# len_y = 0.3   # Height of the cuboid along the y-axis
# len_z = 0.1   # Width of the cuboid along the z-axis
# spacing = 2.0/64/2  # Spacing between particles
# random_disturbance_mag = 0.8  # Random disturbance magnitude
#
# # Generate the particles
# particles = fill_cuboid_with_particles(len_x, len_y, len_z, spacing)
# visualize_particles(particles.detach().numpy())
# a=1

