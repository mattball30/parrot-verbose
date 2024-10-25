import argparse
import os
import pickle

import pandas as pd
from tqdm import tqdm

BOS, EOS, PAD, MASK = "[BOS]", "[EOS]", "[PAD]", "[MASK]"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converts reaction csv to idx and tokens"
    )
    parser.add_argument(
        "--source_data_path",
        type=str,
        default="../../dataset/source_dataset/",
        help="Path to source data",
    )
    parser.add_argument(
        "--dataset_dir_name",
        type=str,
        help="Path to final condition data",
    )
    parser.add_argument(
        "--dataset_fname",
        type=str,
        help="Final condition data file name",
    )
    parser.add_argument(
        "--idx2data_fpath",
        type=str,
        default=None,
        help="Path to idx2data mapping file",
    )
    parser.add_argument(
        "--calculate_fps",
        action="store_true",
        help="Calculate fps for all data",
    )
    return parser.parse_args()


def get_condition2idx_mapping(all_condition_data: pd.DataFrame):
    col_unique_data = [BOS, EOS, PAD, MASK]
    for col in all_condition_data.columns.tolist():
        one_col_unique = list(set(all_condition_data[col].tolist()))
        col_unique_data.extend(one_col_unique)
    col_unique_data = list(set(col_unique_data))
    col_unique_data = [x if x is not None else "" for x in col_unique_data]
    col_unique_data.sort()
    idx2data = {i: x for i, x in enumerate(col_unique_data)}
    data2idx = {x: i for i, x in enumerate(col_unique_data)}
    return idx2data, data2idx


if __name__ == "__main__":
    args = parse_args()

    calculate_fps = args.calculate_fps

    debug = False
    fp_size = 16384
    source_data_path = args.source_data_path
    final_condition_data_path = os.path.join(source_data_path, args.dataset_dir_name)
    database_fname = args.dataset_fname

    if debug:
        database = pd.read_csv(
            os.path.join(final_condition_data_path, database_fname), nrows=10000
        )
        final_condition_data_path = os.path.join(
            source_data_path, "USPTO_condition_final_debug"
        )
        if not os.path.exists(final_condition_data_path):
            os.makedirs(final_condition_data_path)
        database.to_csv(
            os.path.join(final_condition_data_path, database_fname), index=False
        )
    else:
        database = pd.read_csv(os.path.join(final_condition_data_path, database_fname))
        database = database.fillna("")

    condition_cols = ["catalyst1", "solvent1", "solvent2", "reagent1", "reagent2"]

    if not args.idx2data_fpath:
        all_idx2data, all_data2idx = get_condition2idx_mapping(database[condition_cols])
        all_idx_mapping_data_fpath = os.path.join(
            final_condition_data_path,
            "{}_alldata_idx.pkl".format(database_fname.split(".")[0]),
        )
        with open(all_idx_mapping_data_fpath, "wb") as f:
            pickle.dump((all_idx2data, all_data2idx), f)
    else:
        with open(args.idx2data_fpath, "rb") as f:
            all_idx2data, all_data2idx = pickle.load(f)

    all_condition_labels = []
    for _, row in tqdm(database[condition_cols].iterrows(), total=len(database)):
        row = list(row)
        row = ["[BOS]"] + row + ["[EOS]"]
        all_condition_labels.append([all_data2idx[x] for x in row])

    all_condition_labels_fpath = os.path.join(
        final_condition_data_path,
        "{}_condition_labels.pkl".format(database_fname.split(".")[0]),
    )
    with open(all_condition_labels_fpath, "wb") as f:
        pickle.dump((all_condition_labels), f)
    print("Done!")
