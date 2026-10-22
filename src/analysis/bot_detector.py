"""ML-based Bot and BuzzerRp detection via Semantic Clustering.

Uses IndoBERTweet to embed comments and DBSCAN to cluster them,
catching 'spin-text' bots that use variations of the same talking point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Force UTF-8 encoding for standard output to avoid Windows console errors
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def detect_bots(
    parquet_path: Path,
    min_authors: int = 5,
    top_n: int = 20,
    eps: float = 0.15,
) -> None:
    if not parquet_path.is_file():
        print(f"ERROR: File not found {parquet_path}. Run ingest and normalize first.", file=sys.stderr)
        return

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    if "text_normalized" not in df.columns or "author_hash" not in df.columns:
        print(f"ERROR: Missing required columns in {parquet_path}.", file=sys.stderr)
        return
    
    # Filter short comments
    df = df[df["text_normalized"].str.len() >= 20].copy()
    print(f"Analyzing {len(df)} valid comments for semantic copypastas...")

    # Deduplicate text to save embedding time
    unique_texts = df["text_normalized"].unique()
    print(f"Found {len(unique_texts)} unique texts. Loading IndoBERTweet model...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        print(f"ERROR: ML libraries missing. {e}", file=sys.stderr)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "indolem/indobertweet-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Compute embeddings in batches
    batch_size = 64
    all_embeddings = []
    
    print(f"Computing embeddings in batches of {batch_size}...")
    with torch.no_grad():
        for i in range(0, len(unique_texts), batch_size):
            batch_texts = unique_texts[i:i+batch_size].tolist()
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings for cosine similarity equivalence
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i}/{len(unique_texts)} texts")

    emb_matrix = np.vstack(all_embeddings)

    print("Clustering embeddings using DBSCAN...")
    # eps=0.15 on L2-normalized vectors implies very high cosine similarity requirement
    clustering = DBSCAN(eps=eps, min_samples=2, metric="euclidean").fit(emb_matrix)
    
    # Map clusters back to unique texts
    text_to_cluster = dict(zip(unique_texts, clustering.labels_))
    
    # Apply to dataframe
    df["semantic_cluster_id"] = df["text_normalized"].map(text_to_cluster)
    
    # -1 is noise in DBSCAN
    df = df[df["semantic_cluster_id"] != -1].copy()
    print(f"Found {df['semantic_cluster_id'].nunique()} distinct semantic campaigns.")
    
    # Group by cluster ID
    grouped = df.groupby("semantic_cluster_id").agg(
        unique_authors_count=("author_hash", "nunique"),
        unique_videos_count=("post_id", "nunique"),
        total_likes=("like_count", "sum"),
        # Get the most common text in the cluster as the representative
        representative_text=("text_normalized", lambda x: x.value_counts().index[0])
    ).reset_index()
    
    # Filter by minimum unique authors
    bot_clusters = grouped[grouped["unique_authors_count"] >= min_authors].copy()
    
    if bot_clusters.empty:
        print(f"\nNo coordinated copypastas found (min_authors={min_authors}).")
        return
    
    # Sort
    bot_clusters = bot_clusters.sort_values(
        by=["unique_videos_count", "total_likes", "unique_authors_count"], 
        ascending=[False, False, False]
    ).head(top_n)
    
    print("\n" + "="*120)
    print(f"TOP SUSPECTED BUZZER CAMPAIGNS (Top {len(bot_clusters)})")
    print("="*120)
    
    for i, row in bot_clusters.iterrows():
        authors = row['unique_authors_count']
        videos = row['unique_videos_count']
        likes = row['total_likes']
        
        # Find variants
        cluster_df = df[df["semantic_cluster_id"] == row["semantic_cluster_id"]]
        variants = cluster_df["text_normalized"].unique()
        
        print(f"[{authors} authors | {videos} videos | {likes} likes] -> {row['representative_text']}")
        if len(variants) > 1:
            print(f"  + {len(variants)-1} semantic variants (e.g., '{variants[1]}')")
        
    print("="*120)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect coordinated bot/buzzer activity using ML Semantic Clustering.",
    )
    parser.add_argument(
        "--input", 
        default="data/interim/comments_normalized.parquet",
        help="Path to the normalized comments parquet file."
    )
    parser.add_argument(
        "--min-authors",
        type=int,
        default=5,
        help="Minimum number of unique authors posting the exact same text to be flagged."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top clusters to display."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.15,
        help="DBSCAN epsilon parameter for embedding distance."
    )
    args = parser.parse_args(argv)
    
    detect_bots(
        parquet_path=Path(args.input),
        min_authors=args.min_authors,
        top_n=args.top,
        eps=args.eps,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
