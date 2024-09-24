"""Combines batched datasets into a single dataset."""

#! /usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
import tqdm

CONDITIONS_TO_COMBINE = ["catalyst1", "reagent1", "reagent2", "solvent1", "solvent2"]


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: Command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Combine batched datasets into a single dataset."
    )
    parser.add_argument(
        "--input_dir", help="Directory containing batched datasets.", required=True
    )
    parser.add_argument(
        "--num_batches", help="Number of batches to combine.", required=True, type=int
    )
    parser.add_argument(
        "--task_name",
        help="Name of the task which generated the batched dataset files: <task_name>_batch_<n>_reagent1.pkl etc.",
        required=True,
        type=str,
    )

    return parser.parse_args()


def append_array_to_file(file_path: str, data: np.array, append_mode="ab") -> None:
    """Append array data to file using binary write.

    :param file_path: Path to the file to append data to.
    :type file_path: str
    :param data: Data to append to the file.
    :type data: np.array
    :param append_mode: Mode to open the file in, defaults to "ab".
    :type append_mode: str, optional
    """
    with open(file_path, append_mode) as f:
        np.save(f, data)


def combine_batched_datasets(input_dir: str, task_name: str, num_batches: int) -> None:
    """Combines batched datasets into a single dataset.

    :param input_dir: Input directory containing batched datasets.
    :type input_dir: str
    :param task_name: Name of the task which generated the batched dataset files: <task_name>_batch_<n>_reagent1.pkl etc.
    :type task_name: str
    :param num_batches: Number of batches to combine.
    :type num_batches: int
    """
    final_rxn_fps_path = f"{input_dir}/{task_name}_rxn_fps.npy"
    final_prod_fps_path = f"{input_dir}/{task_name}_prod_fps.npy"

    # Remove the files if they already exist, to avoid appending to an old file
    if os.path.exists(final_rxn_fps_path):
        os.remove(final_rxn_fps_path)
    if os.path.exists(final_prod_fps_path):
        os.remove(final_prod_fps_path)

    print("Appending rxn_fps data incrementally")
    for i in range(num_batches):
        rxn_fps = np.load(
            f"{input_dir}/{task_name}_batch_{i}_rxn_fps.npz", mmap_mode="r"
        )["fps"]
        append_array_to_file(final_rxn_fps_path, rxn_fps)

    print("Appending prod_fps data incrementally")
    for i in range(num_batches):
        prod_fps = np.load(
            f"{input_dir}/{task_name}_batch_{i}_prod_fps.npz", mmap_mode="r"
        )["fps"]
        append_array_to_file(final_prod_fps_path, prod_fps)

    # After appending, you will have two `.npy` files that contain all the concatenated data
    print(f"Final rxn_fps saved to {final_rxn_fps_path}")
    print(f"Final prod_fps saved to {final_prod_fps_path}")

    # Combine conditions.pkl
    print("Combining conditions")
    for condition in tqdm.tqdm(CONDITIONS_TO_COMBINE):
        combined_idx_to_cond = {}
        combined_cond_to_idx = {}
        for i in range(num_batches):
            with open(
                f"{input_dir}/{task_name}_batch_{i}_{condition}.pkl", "rb"
            ) as file:
                idx_to_cond, cond_to_idx = pickle.load(file)
                combined_idx_to_cond.update(idx_to_cond)
                combined_cond_to_idx.update(cond_to_idx)

        conditions = (combined_idx_to_cond, combined_cond_to_idx)
        with open(f"{input_dir}/{task_name}_{condition}.pkl", "wb") as f:
            pickle.dump(conditions, f)

        print(f"Saved {condition}.pkl")

    return


def main(args: argparse.Namespace):
    """Combines batched dataset information for RCR into a single dataset.

    In more detail, this combines the *_rxn_fps.npz, *_prod_fps.npz, *_{condition}.pkl files
    into a single rxn_fps.npz, prod_fps.npz, and {condition}.pkl file.

    :param args: Command-line arguments.
    :type args: argparse.Namespace
    """
    input_dir = args.input_dir
    task_name = args.task_name
    num_batches = args.num_batches

    combine_batched_datasets(input_dir, task_name, num_batches)


if __name__ == "__main__":
    args = parse_args()

    main(args)
