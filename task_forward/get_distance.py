import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import seed_everything

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Search for similar reactions.")
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input data.",
    )
    parser.add_argument(
        "--target_embedding",
        type=str,
        required=True,
        help="Path to the target embedding.",
    )
    parser.add_argument(
        "--query_embedding",
        type=str,
        required=True,
        help="Path to the target embedding.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory where results are saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    seed_everything(42)

    target_embedding = np.load(config.target_embedding)
    query_embedding = np.load(config.query_embedding)

    target_embedding = torch.tensor(target_embedding, dtype=torch.float32).cuda()
    query_embedding = torch.tensor(query_embedding, dtype=torch.float32).cuda()

    target_embedding = torch.nn.functional.normalize(target_embedding, p=2, dim=1)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    batch_size = config.batch_size
    distances = []

    for i in range(0, query_embedding.shape[0], batch_size):
        print(f"Processing batch {i // batch_size}...")
        batch = query_embedding[i : i + batch_size]
        similarity = torch.matmul(batch, target_embedding.T)
        distance, _ = torch.max(similarity, dim=1)
        distances.append(distance.cpu().tolist())

    distances = np.concatenate(distances)

    df = pd.read_csv(config.input_data)
    df["distance"] = distances
    df.to_csv(os.path.join(config.output_dir, "distance.csv"), index=False)
