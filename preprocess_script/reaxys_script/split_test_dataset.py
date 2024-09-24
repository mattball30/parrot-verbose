#! /usr/bin/env python

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../../baseline_model/")
from baseline_condition_model import create_rxn_Morgan2FP_separately


def get_idx_dict(data):
    unique_data = list(set(data))
    unique_data.sort()
    idx2data = {i: x for i, x in enumerate(unique_data)}
    data2idx = {x: i for i, x in enumerate(unique_data)}
    return idx2data, data2idx


def proprecess_baseline_dataset(
    database: pd.DataFrame, final_condition_data_path, database_fname
):
    canonical_rxn = database.canonical_rxn.tolist()
    prod_fps = []
    rxn_fps = []
    for rxn in tqdm(canonical_rxn):
        rsmi, psmi = rxn.split(">>")
        [pfp, rfp] = create_rxn_Morgan2FP_separately(
            rsmi,
            psmi,
            rxnfpsize=fp_size,
            pfpsize=fp_size,
            useFeatures=False,
            calculate_rfp=True,
            useChirality=True,
        )
        rxn_fp = pfp - rfp
        prod_fps.append(pfp)
        rxn_fps.append(rxn_fp)

    print("Converting fingerprints to numpy arrays")
    prod_fps = np.array(prod_fps)
    rxn_fps = np.array(rxn_fps)

    # Convert final_condition_data_path to Path object
    final_condition_data_path = os.path.join(final_condition_data_path)
    full_fps_path = os.path.join(
        final_condition_data_path, f'{database_fname.split(".")[0]}_prod_fps'
    )
    print(f"Saving product fingerprints to {full_fps_path}")
    np.savez_compressed(
        os.path.join(
            final_condition_data_path,
            "{}_prod_fps".format(database_fname.split(".")[0]),
        ),
        fps=prod_fps,
    )
    np.savez_compressed(
        os.path.join(
            final_condition_data_path,
            "{}_rxn_fps".format(database_fname.split(".")[0]),
        ),
        fps=rxn_fps,
    )
    condition_cols = ["catalyst1", "solvent1", "solvent2", "reagent1", "reagent2"]
    for col in condition_cols:
        database[col][pd.isna(database[col])] = ""
        fdata = database[col]
        fpath = os.path.join(
            final_condition_data_path,
            "{}_{}.pkl".format(database_fname.split(".")[0], col),
        )
        print("Saving index dictionaries")
        with open(fpath, "wb") as f:
            pickle.dump(get_idx_dict(fdata.tolist()), f)

    # Check whether the dataset exists first:
    if os.path.exists(os.path.join(final_condition_data_path, database_fname)):
        print("File already exists, skipping")
        return

    else:
        print("Saving dataset to csv")
        database.to_csv(
            os.path.join(final_condition_data_path, database_fname), index=False
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_size", type=int, default=16384, help="fingerprint size")
    parser.add_argument(
        "--source_data_path",
        type=str,
        default="/data/mball/data/uspto-rcr/USPTO_condition.csv",
        help="The path to the source data",
    )
    parser.add_argument(
        "--final_condition_data_path",
        type=str,
        default="/data/mball/data/uspto-rcr/",
        help="The directory where the final data will be saved",
    )
    parser.add_argument(
        "--database_fname",
        type=str,
        default="uspto_condition_test.csv",
        help="Name of the files to save the fingerprints and indicies",
    )
    parser.add_argument(
        "--split_catalyst",
        type=bool,
        default=False,
        help="Split the dataset by presence of catalyst",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Name of the dataset to preprocess, if left empty, preprocess all datasets",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="Number of batches to split and preprocess the dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fp_size = args.fp_size
    source_data_path = args.source_data_path
    final_condition_data_path = args.final_condition_data_path
    database_fname = args.database_fname
    split_catalyst = args.split_catalyst
    dataset_str = args.dataset
    num_batches = args.num_batches

    database = pd.read_csv(source_data_path)

    if len(dataset_str) == 0:
        dataset = database
    else:
        dataset = database.loc[database["dataset"] == dataset_str]

    if not split_catalyst:
        # Partition the pandas dataset into num_batches, preserving the index:
        batched_datasets = np.array_split(dataset, num_batches)
        for i, batch in tqdm(enumerate(batched_datasets)):
            print(f"Processing batch {i}")
            proprecess_baseline_dataset(
                batch,
                final_condition_data_path=final_condition_data_path,
                database_fname=f"{database_fname.split('.')[0]}_batch_{i}.csv",
            )

    else:
        have_catalyst_test_datset = dataset.loc[
            ~dataset["catalyst1"].isna()
        ].reset_index(drop=True)
        print("Include Catalyst dataset # {}".format(len(have_catalyst_test_datset)))

        proprecess_baseline_dataset(
            have_catalyst_test_datset,
            final_condition_data_path=final_condition_data_path,
            database_fname="Reaxys_total_syn_condition_test_have_catalyst.csv",
        )

        catalyst_na_test_datset = dataset.loc[dataset["catalyst1"].isna()].reset_index(
            drop=True
        )
        print("Include Catalyst dataset # {}".format(len(catalyst_na_test_datset)))

        proprecess_baseline_dataset(
            catalyst_na_test_datset,
            final_condition_data_path=final_condition_data_path,
            database_fname="Reaxys_total_syn_condition_test_catalyst_na.csv",
        )
        print("Catalyst na dataset # {}".format(len(catalyst_na_test_datset)))
