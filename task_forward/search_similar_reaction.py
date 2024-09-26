import os
import warnings
import pandas as pd
import numpy as np
import torch
import argparse
import sys

sys.path.append("../")
from utils import seed_everything

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for similar reactions."
    )
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of similar reactions to retrieve.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=-1.0,
        help="Threshold of similarity to retrieve.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size."
    )
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

    embedding = torch.nn.functional.normalize(target_embedding, p=2, dim=1)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    batch_size = config.batch_size
    nearest_samples = []
    similarities = []

    for i in range(0, query_embedding.shape[0], batch_size):
        print(f"Processing batch {i//batch_size}...")
        batch = query_embedding[i: i + batch_size]
        similarity = torch.matmul(batch, embedding.T)
        similarity, nearest_sample = torch.topk(similarity, k=config.top_k, dim=1)
        nearest_samples.append(nearest_sample.cpu().tolist())
        similarities.append(similarity.cpu().tolist())
    nearest_samples = np.concatenate(nearest_samples).flatten()
    similarities = np.concatenate(similarities).flatten()

    if config.similarity_threshold > 0:
        mask = similarities > config.similarity_threshold
        nearest_samples = nearest_samples[mask]

    nearest_samples = set(nearest_samples)
    
    df = pd.read_csv(config.input_data)
    df = df.iloc[list(nearest_samples)]
    df.to_csv(os.path.join(config.output_dir, f"top{config.top_k}_threshold{str(config.similarity_threshold).replace('.', '')}_similar_reactions.csv"), index=False)