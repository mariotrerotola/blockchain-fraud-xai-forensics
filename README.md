# Blockchain Fraud Detection for Explainable Forensic Investigations

Code accompanying the paper of the same title by Mario Trerotola, Davide Calvaresi
and Mimmo Parente (Springer, 2026).

We classify Bitcoin wallets and ERC-20 tokens as fraudulent or licit from
graph and temporal features of their transactions, train imbalance-aware
ensembles, and explain the predictions with SHAP, CIU and DEXiRE. The two
ledgers share the modelling and explanation code; only the feature extraction
is chain specific.

The repository contains code only. The label sources cannot be redistributed,
so you have to rebuild the data yourself before running anything; see
[Data](#data).


## Contents

```
src/bfx/           Python package used by the notebooks
notebooks/bitcoin/ Bitcoin case study, Section 3 of the paper
notebooks/erc20/   ERC-20 case study, Section 4 of the paper
data/README.md     where each input file comes from and where to put it
docs/protocol.md   implementation choices (splits, grids, thresholds)
requirements.txt   pinned dependencies
```

### `src/bfx`

- `bitcoin.py`: download and features for Bitcoin wallets. `fetch_wallet` saves the
  blockchain.info `rawaddr` response of an address as JSON;
  `sample_recent_addresses` collects addresses from the unconfirmed-transaction
  pool; `wallet_features` turns the transactions of one wallet into the 21
  features listed in `FEATURES`; `features_from_json_dir` does the same for a
  whole folder.

- `erc20.py`: download and features for ERC-20 tokens. `fetch_token_transfers` writes the
  first 500 transfers of a contract (ERC-20, native and internal ETH, oldest
  first) from the Etherscan v2 API; `read_transfers` loads such a file;
  `transfer_graph` builds the address graph; `token_features` returns the 146
  numeric descriptors of a token.

- `modeling.py`: code shared by both case studies. `harmonise` merges labelled sources with the
  rule "fraud if any source says fraud"; `scores` returns the metrics we report;
  `ciu_means` averages CIU importance and utility over a set of instances;
  `distil_and_extract` trains a small MLP on the predictions of a forest and runs
  DEXiRE on it; `readable_rule` rewrites DEXiRE thresholds in the original units
  of the features.

### Notebooks

Run the notebooks of a folder in numerical order. Each one reads what the
previous one wrote.

| Notebook | Reads | Writes |
|---|---|---|
| `bitcoin/01_data_collection` | `data/bitcoin/labels/` | `data/bitcoin/transactions/`, `data/bitcoin/features/` |
| `bitcoin/02_dataset_construction` | `data/bitcoin/features/` | `data/bitcoin/scenarios/scenario_{1,2,3}.csv` |
| `bitcoin/03_classification` | scenarios | `results/bitcoin/table3_*`, `models/bitcoin/*.joblib` |
| `bitcoin/04_confident_learning` | scenario 2, RF model | `results/bitcoin/confident_learning.json` |
| `bitcoin/05_explainability` | scenario 2, RF model | `results/bitcoin/table5_shap.csv`, `fig1_shap_beeswarm.png`, `ciu_means.csv`, `dexire*` |
| `erc20/01_data_collection` | `data/erc20/labels/` | `data/erc20/transfers/`, `data/erc20/features/` |
| `erc20/02_collinearity_pruning` | features | `results/erc20/selected_features.csv`, `pruning_log.csv`, `pruning_curve.png` |
| `erc20/03_source_scenarios` | features, selected features | `results/erc20/table8_*.csv`, `models/erc20/*.joblib` |
| `erc20/04_explainability` | CA + CMC model | `results/erc20/fig2_shap_beeswarm.png`, `table9_shap_ciu.csv`, `dexire*` |

The two `01` notebooks only download when you set `DOWNLOAD = True` in their
first cell. With the flag off they just compute features from whatever is
already in `data/`.


## Setup

We used Python 3.12 on macOS; Linux works the same way.

```bash
git clone https://github.com/mariotrerotola/blockchain-fraud-xai-forensics.git
cd blockchain-fraud-xai-forensics
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

DEXiRE is installed from GitHub at a fixed commit (last line of
`requirements.txt`) and pulls in TensorFlow. If you only need the
classification results you can skip it; the DEXiRE cells are the last ones of
the two explainability notebooks.

The notebooks add `src/` to the import path themselves, so there is nothing to
install with `pip install -e`.


## Data

`data/README.md` lists every input file, its source and the folder it goes in.
In short:

* Bitcoin needs the ChainAbuse report export, `wallets_classes.csv` from the
  Elliptic++ release, and (optionally) a fresh sample of recent addresses.
  Transaction histories are then downloaded with `bitcoin/01_data_collection`.
* ERC-20 needs four token lists (TokenScout scam and legitimate, ChainAbuse,
  CoinMarketCap). Transfers are downloaded with `erc20/01_data_collection`,
  which reads the Etherscan key from the `ETHERSCAN_API_KEY` environment
  variable:

  ```bash
  export ETHERSCAN_API_KEY=your-key
  ```

Expect the downloads to be the slow part: blockchain.info and Etherscan are
rate limited, and there are tens of thousands of wallets and over 150,000 tokens
to fetch. Both download loops skip files that are already on disk, so they can be
stopped and restarted.


## Running

Start Jupyter from the repository root and open the notebooks:

```bash
jupyter lab
```

or run them headless, one folder at a time:

```bash
cd notebooks/bitcoin
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

Once the feature tables exist, the modelling notebooks take a few minutes each
on a laptop. The longest are the grid searches in `bitcoin/03` and
`erc20/03` and the CIU loops in the explainability notebooks (200 instances
each). All random seeds are fixed to 42. TensorFlow is not bit-for-bit
deterministic on CPU, so the DEXiRE figures can move slightly between runs.

`results/` and `models/` are created by the notebooks and are not tracked.


## Citation

Please cite the paper if you use this code; `CITATION.cff` has the details.


## License

MIT, see `LICENSE`. DEXiRE is a separate project (AISLab, HES-SO) with its own
MIT license.
