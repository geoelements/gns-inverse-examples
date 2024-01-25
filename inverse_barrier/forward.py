import torch
from tqdm import tqdm
import os
import sys
sys.path.append('gns')
from gns import learned_simulator


KINEMATIC_PARTICLE = 6
STATIONARY_PARTICLE = 3


def rollout_with_checkpointing(
        simulator: learned_simulator.LearnedSimulator,
        initial_positions: torch.tensor,
        particle_types: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        checkpoint_interval: int = 1,
        material_property: torch.tensor = None,
        knwon_positions: torch.tensor = None
):
    """ Rollout with gradient checkpointing to reduce memory accumulation over the forward steps during backpropagation.
    Args:
      simulator: learned_simulator
      initial_positions: initial positions of particles for 6 timesteps with shape=(nparticles, 6, ndims).
      particle_types: particle types shape=(nparticles, ).
      n_particles_per_example: number of particles.
      nsteps: number of forward steps to rollout.
      checkpoint_interval: frequency of gradient checkpointing.
      material_property: Friction angle normalized by tan() with shape (nparticles, )
    Returns:
      GNS rollout of particles positions
    """

    current_positions = initial_positions
    predictions = []

    for step in tqdm(range(nsteps), total=nsteps):
        if step % checkpoint_interval == 0:
            next_position = torch.utils.checkpoint.checkpoint(
                simulator.predict_positions,
                current_positions,
                [n_particles_per_example],
                particle_types,
                material_property
            )
        else:
            next_position = simulator.predict_positions(
                current_positions,
                [n_particles_per_example],
                particle_types,
                material_property
            )

        stationary_mask = (particle_types == STATIONARY_PARTICLE).clone().detach()
        stationary_mask = stationary_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
        next_position = torch.where(
            stationary_mask, knwon_positions, next_position)
        predictions.append(next_position)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1)

    return torch.cat(
        (initial_positions.permute(1, 0, 2), torch.stack(predictions))
    )
