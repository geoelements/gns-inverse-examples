import torch
import torch.utils.checkpoint
from tqdm import tqdm


def rollout_with_checkpointing(
        simulator,
        initial_positions: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        checkpoint_interval=5,
        is_checkpointing=True
        ):


    current_positions = initial_positions
    predictions = []

    if is_checkpointing:
        for step in tqdm(range(nsteps), total=nsteps):
            # print(f"Step {step}/{nsteps}")
            if step % checkpoint_interval == 0:  # Checkpoint every 2 time steps
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
                    nparticles_per_example=[n_particles_per_example],
                    particle_types=particle_types,
                    material_property=material_property
                )

            predictions.append(next_position)

            # Shift `current_positions`, removing the oldest position in the sequence
            # and appending the next position at the end.
            current_positions = torch.cat(
                [current_positions[:, 1:], next_position[:, None, :]], dim=1)

    else:
        for step in tqdm(range(nsteps), total=nsteps):
            next_position = simulator.predict_positions(
                current_positions,
                nparticles_per_example=[n_particles_per_example],
                particle_types=particle_types,
                material_property=material_property
            )
            predictions.append(next_position)

            # Shift `current_positions`, removing the oldest position in the sequence
            # and appending the next position at the end.
            current_positions = torch.cat(
                [current_positions[:, 1:], next_position[:, None, :]], dim=1)

    return torch.cat(
        (initial_positions.permute(1, 0, 2), torch.stack(predictions))
    )

