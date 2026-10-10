"""ERC-20 token features.

For each token contract the first transfers returned by Etherscan (ERC-20
transfers, native ETH transfers and internal transfers to or from the contract,
in chronological order) are turned into a directed graph whose nodes are
addresses and whose edges aggregate the transfers between two addresses.
Four families of descriptors are computed on the graph and on the raw
transfer list; node- and edge-level descriptors are summarised by their mean,
standard deviation, minimum and maximum.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests

ETHERSCAN_URL = "https://api.etherscan.io/v2/api"
PAGE_SIZE = 1_000
TRANSFER_FIELDS = ["transaction_hash", "timestamp", "from_address", "to_address", "value",
                   "token_symbol", "transaction_type"]
STATS = ("mean", "std", "min", "max")


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _etherscan(params: dict, api_key: str, pause: float = 0.22) -> list[dict]:
    time.sleep(pause)
    payload = requests.get(ETHERSCAN_URL, params={**params, "apikey": api_key, "chainid": 1},
                           timeout=30).json()
    result = payload.get("result", [])
    return result if isinstance(result, list) else []


def _scaled(raw: str, decimals: int) -> float:
    try:
        return int(raw) / 10 ** decimals
    except (TypeError, ValueError):
        return 0.0


def fetch_token_transfers(contract: str, out_path: Path, api_key: str | None = None,
                          max_transfers: int = 500) -> int:
    """Write up to ``max_transfers`` transfers of ``contract`` to ``out_path``."""
    api_key = api_key or os.environ["ETHERSCAN_API_KEY"]
    actions = [("tokentx", "contractaddress", "ERC20_TRANSFER"),
               ("txlist", "address", "ETH_NATIVE"),
               ("txlistinternal", "address", "ETH_INTERNAL")]
    rows: list[dict] = []
    for action, key, kind in actions:
        page = 1
        while len(rows) < max_transfers:
            batch = _etherscan({"module": "account", "action": action, key: contract,
                                "page": page, "offset": PAGE_SIZE, "sort": "asc"}, api_key)
            for tx in batch:
                decimals = int(tx.get("tokenDecimal") or 18)
                value = _scaled(tx.get("value", "0"), decimals)
                if kind != "ERC20_TRANSFER" and value <= 0:
                    continue
                rows.append({
                    "transaction_hash": tx.get("hash", ""),
                    "timestamp": datetime.fromtimestamp(int(tx["timeStamp"])).strftime("%Y-%m-%d %H:%M:%S"),
                    "from_address": tx.get("from", "").lower(),
                    "to_address": tx.get("to", "").lower(),
                    "value": value,
                    "token_symbol": tx.get("tokenSymbol", "ETH"),
                    "transaction_type": kind,
                })
            if len(batch) < PAGE_SIZE:
                break
            page += 1
    rows = rows[:max_transfers]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRANSFER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_transfers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df["from_address"] = df["from_address"].astype(str).str.lower()
    df["to_address"] = df["to_address"].astype(str).str.lower()
    return df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------
# Graph and descriptors
# --------------------------------------------------------------------------

def transfer_graph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for row in df.itertuples(index=False):
        u, v, value, ts = row.from_address, row.to_address, float(row.value), row.timestamp
        if g.has_edge(u, v):
            d = g[u][v]
            d["weight"] += value
            d["count"] += 1
            d["timestamps"].append(ts)
            d["values"].append(value)
        else:
            g.add_edge(u, v, weight=value, count=1, timestamps=[ts], values=[value])
    return g


def _stats(values) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return dict.fromkeys(STATS, 0.0)
    return {"mean": arr.mean(), "std": arr.std(), "min": arr.min(), "max": arr.max()}


def _put(features: dict, name: str, values) -> None:
    for stat, value in _stats(values).items():
        features[f"{name}_{stat}"] = float(value)


def _centralities(g: nx.DiGraph, features: dict, betweenness_k: int = 100) -> None:
    funcs = {
        "node_degree_centrality": nx.degree_centrality,
        "node_indegree_centrality": nx.in_degree_centrality,
        "node_outdegree_centrality": nx.out_degree_centrality,
        "node_betweenness_centrality": lambda x: nx.betweenness_centrality(x, k=min(betweenness_k, len(x))),
        "node_closeness_centrality": nx.closeness_centrality,
        "node_eigenvector_centrality": lambda x: nx.eigenvector_centrality(x, max_iter=1000, tol=1e-6),
        "node_katz_centrality": lambda x: nx.katz_centrality(x, max_iter=1000, tol=1e-6),
        "node_clustering_coefficient": lambda x: nx.clustering(x.to_undirected()),
    }
    for name, func in funcs.items():
        try:
            _put(features, name, func(g).values())
        except Exception:  # power iterations may not converge on degenerate graphs
            _put(features, name, [])


def _node_temporal(g: nx.DiGraph, df: pd.DataFrame, features: dict) -> None:
    median_ts = df["timestamp"].median()
    buckets: dict[str, list[float]] = {}
    for node in g.nodes():
        for kind, edges in (("incoming", g.in_edges(node, data=True)), ("outgoing", g.out_edges(node, data=True))):
            edges = list(edges)
            long_ = [d for _, _, d in edges if d["timestamps"][-1] < median_ts]
            short = [d for _, _, d in edges if d["timestamps"][-1] >= median_ts]
            for horizon, sel in (("long", long_), ("short", short)):
                buckets.setdefault(f"node_{horizon}_term_{kind}_count", []).append(len(sel))
                buckets.setdefault(f"node_{horizon}_term_{kind}_avg_value", []).append(
                    float(np.mean([d["weight"] for d in sel])) if sel else 0.0)
    for name, values in buckets.items():
        _put(features, name, values)


def _edge_metrics(g: nx.DiGraph, df: pd.DataFrame, features: dict) -> None:
    total = df["value"].sum()
    median_ts, max_ts = df["timestamp"].median(), df["timestamp"].max()
    b: dict[str, list[float]] = {k: [] for k in (
        "edge_transfer_value", "edge_transfer_value_normalized", "edge_transfer_value_log",
        "edge_harmonic_transfer_value", "edge_accumulated_incoming_value", "edge_accumulated_incoming_count",
        "edge_accumulated_outgoing_value", "edge_accumulated_outgoing_count",
        "edge_long_term_accum_incoming_freq", "edge_short_term_accum_incoming_freq",
        "edge_long_term_accum_outgoing_freq", "edge_short_term_accum_outgoing_freq",
        "edge_transfer_frequency", "edge_transfer_recency")}
    for u, v, d in g.edges(data=True):
        w = d["weight"]
        b["edge_transfer_value"].append(w)
        b["edge_transfer_value_normalized"].append(w / total if total > 0 else 0.0)
        b["edge_transfer_value_log"].append(float(np.log(w + 1)) if w > 0 else 0.0)
        b["edge_harmonic_transfer_value"].append(1 / (w + 1) if w > 0 else 0.0)
        b["edge_accumulated_incoming_value"].append(sum(x["weight"] for _, _, x in g.in_edges(v, data=True)))
        b["edge_accumulated_incoming_count"].append(g.in_degree(v))
        b["edge_accumulated_outgoing_value"].append(sum(x["weight"] for _, _, x in g.out_edges(u, data=True)))
        b["edge_accumulated_outgoing_count"].append(g.out_degree(u))
        b["edge_transfer_frequency"].append(d["count"])
        b["edge_transfer_recency"].append((max_ts - d["timestamps"][-1]).total_seconds() / 86_400)
        n_long = sum(1 for t in d["timestamps"] if t < median_ts)
        n_short = len(d["timestamps"]) - n_long
        b["edge_long_term_accum_incoming_freq"].append(n_long)
        b["edge_short_term_accum_incoming_freq"].append(n_short)
        b["edge_long_term_accum_outgoing_freq"].append(n_long)
        b["edge_short_term_accum_outgoing_freq"].append(n_short)
    for name, values in b.items():
        _put(features, name, values)


def _gini(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    c = np.cumsum(v)
    return float((v.size + 1 - 2 * c.sum() / c[-1]) / v.size)


def _transaction_stats(df: pd.DataFrame, features: dict) -> None:
    v = df["value"]
    features.update({
        "total_transactions": len(df), "total_volume": float(v.sum()),
        "avg_transaction_value": float(v.mean()), "std_transaction_value": float(v.std()),
        "min_transaction_value": float(v.min()), "max_transaction_value": float(v.max()),
        "median_transaction_value": float(v.median()), "q25_transaction_value": float(v.quantile(0.25)),
        "q75_transaction_value": float(v.quantile(0.75)), "q95_transaction_value": float(v.quantile(0.95)),
    })
    senders, receivers = df["from_address"].nunique(), df["to_address"].nunique()
    unique = len(set(df["from_address"]) | set(df["to_address"]))
    features.update({"unique_senders": senders, "unique_receivers": receivers, "unique_addresses": unique})
    features["sender_ratio"] = senders / unique if unique else 0.0
    features["receiver_ratio"] = receivers / unique if unique else 0.0
    features["tx_per_unique_address"] = len(df) / unique if unique else 0.0
    features["avg_volume_per_address"] = features["total_volume"] / unique if unique else 0.0
    features["time_span"] = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86_400
    gaps = df["timestamp"].diff().dt.total_seconds().dropna()
    for stat, value in (("avg", gaps.mean()), ("std", gaps.std()), ("min", gaps.min()), ("max", gaps.max())):
        features[f"{stat}_time_between_tx"] = float(value) if len(gaps) else 0.0
    features["value_concentration"] = _gini(v.values)


def token_features(df: pd.DataFrame) -> dict[str, float]:
    """All descriptors of one token, computed from its transfer list."""
    g = transfer_graph(df)
    features: dict[str, float] = {"num_nodes": g.number_of_nodes(), "num_edges": g.number_of_edges(),
                                  "num_transactions": len(df)}
    if g.number_of_nodes():
        _centralities(g, features)
    _node_temporal(g, df, features)
    _edge_metrics(g, df, features)
    _transaction_stats(df, features)
    return features
