import pathlib
import glob
import re
import json
import h5py
import numpy as np
import os


def convert_hd5_to_npz(path: str, uuid: str, ndim: int, output: str, material_feature: bool, dt=1.0):
    metadata_path = path
    result_path = [path + uuid]  # yc: I temporally force it to be list

    # read ndim if metatdata exists
    if not os.path.exists(f"{metadata_path}/metadata.json"):
        raise FileExistsError(f"The path {metadata_path}/metadata.json does not exist.")
    else:
        # read simulation dimension
        f = open(f"{metadata_path}/metadata.json")
        metadata = json.load(f)
        ndim = len(metadata["simulation_domain"])

    # get all dirs for hd5
    directories = [pathlib.Path(path) for path in result_path]
    for directory in directories:
        if not directory.exists():
            raise FileExistsError(f"The path {directory} does not exist.")
    print(f"Number of trajectories: {len(directories)}")

    # setup up variables to calculate on-line mean and standard deviation
    # for velocity and acceleration.
    ndim = int(ndim)
    if ndim == 2:
        running_sum = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_sumsq = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
        running_count = dict(velocity_x=0, velocity_y=0, acceleration_x=0, acceleration_y=0)
    elif ndim == 3:
        running_sum = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0,
                           acceleration_z=0)
        running_sumsq = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0,
                             acceleration_z=0)
        running_count = dict(velocity_x=0, velocity_y=0, velocity_z=0, acceleration_x=0, acceleration_y=0,
                             acceleration_z=0)
    else:
        raise NotImplementedError

    trajectories = {}
    for nth_trajectory, directory in enumerate(directories):
        fnames = glob.glob(f"{str(directory)}/*.h5")
        get_fnumber = re.compile(".*\D(\d+).h5")
        fnumber_and_fname = [(int(get_fnumber.findall(fname)[0]), fname) for fname in fnames]
        fnumber_and_fname_sorted = sorted(fnumber_and_fname, key=lambda row: row[0])

        # get size of trajectory
        with h5py.File(fnames[0], "r") as f:
            (nparticles,) = f["table"]["coord_x"].shape
        nsteps = len(fnames)

        # allocate memory for trajectory
        # assume number of particles does not change along the rollout.
        positions = np.empty((nsteps, nparticles, ndim), dtype=float)
        print(f"Size of trajectory {nth_trajectory} ({directory}): {positions.shape}")

        # open each file and copy data to positions tensor.
        for nth_step, (_, fname) in enumerate(fnumber_and_fname_sorted):
            with h5py.File(fname, "r") as f:
                for idx, name in zip(range(ndim), ["coord_x", "coord_y", "coord_z"]):
                    positions[nth_step, :, idx] = f["table"][name][:]

        dt = float(dt)
        # compute velocities using finite difference
        # assume velocities before zero are equal to zero
        velocities = np.empty_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / dt
        velocities[0] = 0

        # compute accelerations finite difference
        # assume accelerations before zero are equal to zero
        accelerations = np.empty_like(velocities)
        accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt
        accelerations[0] = 0

        # update variables for on-line mean and standard deviation calculation.
        for key in running_sum:
            if key == "velocity_x":
                data = velocities[:, :, 0]
            elif key == "velocity_y":
                data = velocities[:, :, 1]
            elif key == "velocity_z":
                data = velocities[:, :, 2]
            elif key == "acceleration_x":
                data = accelerations[:, :, 0]
            elif key == "acceleration_y":
                data = accelerations[:, :, 1]
            elif key == "acceleration_z":
                data = accelerations[:, :, 2]
            else:
                raise KeyError

            running_sum[key] += np.sum(data)
            running_sumsq[key] += np.sum(data ** 2)
            running_count[key] += np.size(data)

        if os.path.exists(f"{metadata_path}/metadata.json") and material_feature is True:
            # read material_id and associated material properties in mpm_input.json
            # TODO: currently, it only supports single material type in one simulation
            material_id = metadata["particle"]["group0"]["material_id"]
            f = open(f"{metadata_path}/mpm_input.json")
            mpm_input = json.load(f)
            for material in mpm_input["materials"]:
                if material["id"] == material_id:
                    material_for_id = material
            normalized_friction = np.tan(material_for_id["friction"] * np.pi / 180)

            trajectories[str(directory)] = (
                positions,  # position sequence (timesteps, particles, dims)
                # TODO: particle type is hardcoded to be 6
                np.full(positions.shape[1], 6, dtype=int),  # particle type (particles, )
                np.full(positions.shape[1], normalized_friction, dtype=float))  # material type (particles, )
        else:
            trajectories[str(directory)] = (
                positions,  # position sequence (timesteps, particles, dims)
                np.full(positions.shape[1], 6, dtype=int))   # particle type (particles, )

    # compute online mean and standard deviation.
    print("Statistis across all trajectories:")
    for key in running_sum:
        mean = running_sum[key] / running_count[key]
        std = np.sqrt((running_sumsq[key] - running_sum[key] ** 2 / running_count[key]) / (running_count[key] - 1))
        print(f"  {key}: mean={mean:.4E}, std={std:.4E}")

    np.savez_compressed(output, **trajectories)
    print(f"Output written to: {output}")


