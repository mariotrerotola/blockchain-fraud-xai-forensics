"""Bitcoin wallet features.

A wallet is described by the transactions returned by the blockchain.info
``rawaddr`` endpoint. Every transaction is expanded into (input address,
output address) pairs; per-address statistics are computed on these pairs and
then aggregated to one row per wallet.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAWADDR_URL = "https://blockchain.info/rawaddr/{address}"
UNCONFIRMED_URL = "https://blockchain.info/unconfirmed-transactions?format=json"

EPS = 1e-8

FEATURES = [
    "total_in_degree",
    "total_out_degree",
    "total_unique_in",
    "total_unique_out",
    "avg_in_transaction",
    "avg_out_transaction",
    "total_received",
    "total_sent",
    "net_balance",
    "avg_in_time_interval",
    "avg_out_time_interval",
    "avg_active_duration",
    "in_out_ratio",
    "unique_in_ratio",
    "unique_out_ratio",
    "volume_ratio",
    "net_balance_ratio",
    "activity_index",
    "time_interval_ratio",
    "weighted_avg_tx",
    "wallet_lifetime_sec",
]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def fetch_wallet(address: str, out_dir: Path, session: requests.Session | None = None,
                 retries: int = 3, timeout: int = 30) -> Path | None:
    """Save the rawaddr JSON of ``address`` to ``out_dir``; return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{address}_transactions.json"
    if target.exists():
        return target
    session = session or requests.Session()
    for attempt in range(retries + 1):
        try:
            resp = session.get(RAWADDR_URL.format(address=address), timeout=timeout)
        except requests.RequestException:
            time.sleep(1.5 ** attempt)
            continue
        if resp.status_code == 200:
            target.write_text(resp.text)
            return target
        if resp.status_code not in (429, 500, 502, 503, 504):
            return None
        time.sleep(1.5 ** attempt)
    return None


def sample_recent_addresses(n: int, pause: float = 5.0) -> list[str]:
    """Collect ``n`` distinct addresses from the unconfirmed-transaction pool."""
    seen: dict[str, None] = {}
    while len(seen) < n:
        txs = requests.get(UNCONFIRMED_URL, timeout=30).json().get("txs", [])
        for tx in txs:
            for out in tx.get("out", []):
                if out.get("addr"):
                    seen.setdefault(out["addr"])
        time.sleep(pause)
    return list(seen)[:n]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _pairs(transactions: list[dict]) -> pd.DataFrame:
    records = []
    for tx in transactions:
        ts = datetime.fromtimestamp(tx["time"])
        inputs = [i["prev_out"].get("addr") for i in tx["inputs"] if i.get("prev_out", {}).get("addr")]
        for out in tx["out"]:
            if not out.get("addr"):
                continue
            for in_addr in inputs:
                records.append({"input_address": in_addr, "output_address": out["addr"],
                                "amount": out["value"] / 1e8, "time": ts})
    return pd.DataFrame(records)


def _address_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.sort_values("time").copy()
    pairs["time_diff"] = pairs.groupby("output_address")["time"].diff().dt.total_seconds() / 86_400

    by_out = pairs.groupby("output_address")
    by_in = pairs.groupby("input_address")
    in_deg = by_out.size()
    idx = in_deg.index
    out_deg = by_in.size().reindex(idx, fill_value=0)
    received = by_out["amount"].sum()
    sent = by_in["amount"].sum().reindex(idx, fill_value=0)

    return pd.DataFrame({
        "in_degree": in_deg,
        "out_degree": out_deg,
        "unique_in_degree": by_out["input_address"].nunique(),
        "unique_out_degree": by_in["output_address"].nunique().reindex(idx, fill_value=0),
        "avg_in_transaction": by_out["amount"].mean(),
        "avg_out_transaction": by_in["amount"].mean().reindex(idx, fill_value=0),
        "avg_in_time_interval": by_out["time_diff"].mean(),
        "avg_out_time_interval": by_in["time_diff"].mean().reindex(idx, fill_value=0),
        "balance": received - sent,
        "active_duration": (by_out["time"].max() - by_out["time"].min()).dt.total_seconds() / 86_400,
    })


def wallet_features(transactions: list[dict]) -> dict[str, float] | None:
    """Return the 21 wallet-level features, or ``None`` if the wallet is empty."""
    if not transactions:
        return None
    pairs = _pairs(transactions)
    if pairs.empty:
        return None
    m = _address_metrics(pairs)

    f = {
        "total_in_degree": m["in_degree"].sum(),
        "total_out_degree": m["out_degree"].sum(),
        "total_unique_in": m["unique_in_degree"].sum(),
        "total_unique_out": m["unique_out_degree"].sum(),
        "avg_in_transaction": m["avg_in_transaction"].mean(),
        "avg_out_transaction": m["avg_out_transaction"].mean(),
        "total_received": m["balance"].clip(lower=0).sum(),
        "total_sent": (-m["balance"].clip(upper=0)).sum(),
        "net_balance": m["balance"].sum(),
        "avg_in_time_interval": m["avg_in_time_interval"].mean(),
        "avg_out_time_interval": m["avg_out_time_interval"].mean(),
        "avg_active_duration": m["active_duration"].mean(),
    }
    f["in_out_ratio"] = f["total_in_degree"] / (f["total_out_degree"] + EPS)
    f["unique_in_ratio"] = f["total_unique_in"] / (f["total_in_degree"] + EPS)
    f["unique_out_ratio"] = f["total_unique_out"] / (f["total_out_degree"] + EPS)
    f["volume_ratio"] = f["total_received"] / (f["total_sent"] + EPS)
    f["net_balance_ratio"] = f["net_balance"] / (f["total_received"] + EPS)
    f["activity_index"] = (f["total_in_degree"] + f["total_out_degree"]) / (f["avg_active_duration"] + EPS)
    f["time_interval_ratio"] = f["avg_out_time_interval"] / (f["avg_in_time_interval"] + EPS)
    f["weighted_avg_tx"] = (
        f["avg_in_transaction"] * f["total_in_degree"] + f["avg_out_transaction"] * f["total_out_degree"]
    ) / (f["total_in_degree"] + f["total_out_degree"] + EPS)

    times = [tx["time"] for tx in transactions]
    f["wallet_creation_ts"] = float(min(times))
    f["wallet_last_tx_ts"] = float(max(times))
    f["wallet_lifetime_sec"] = float(max(times) - min(times))
    return {k: float(v) if v is not None else np.nan for k, v in f.items()}


def features_from_json_dir(json_dir: Path) -> pd.DataFrame:
    """Compute features for every ``<address>_transactions.json`` in ``json_dir``."""
    rows = []
    for path in sorted(json_dir.glob("*_transactions.json")):
        try:
            txs = json.loads(path.read_text()).get("txs", [])
        except (json.JSONDecodeError, OSError):
            continue
        feats = wallet_features(txs)
        if feats is not None:
            feats["wallet"] = path.name.split("_")[0]
            rows.append(feats)
    return pd.DataFrame(rows)
