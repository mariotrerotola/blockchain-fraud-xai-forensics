# Data

No data is distributed with this repository. The label sources have their own
terms of use, and the transaction histories can be downloaded again from public
endpoints. This file describes what the notebooks expect and how to rebuild it.

```
data/
├── bitcoin/
│   ├── labels/          chainabuse_reports.csv, wallets_classes.csv, recent_sample.csv
│   ├── transactions/    <source>/<address>_transactions.json
│   ├── features/        <source>_metrics.csv          (written by notebook 01)
│   └── scenarios/       scenario_{1,2,3}.csv          (written by notebook 02)
└── erc20/
    ├── labels/          tokenscout_scam.csv, chainabuse_scam.csv, cmc_licit.csv, tokenscout_licit.csv
    ├── transfers/       <list>/<name>_<contract>.csv
    └── features/        <list>.csv                    (written by notebook 01)
```

## Bitcoin

| File | Source | How to obtain |
|---|---|---|
| `labels/chainabuse_reports.csv` | [ChainAbuse](https://www.chainabuse.com) Bitcoin scam reports | Export the reports for the BTC chain; one row per report with the reported string in an `address` column. |
| `labels/wallets_classes.csv` | [Elliptic++](https://github.com/git-disl/EllipticPlusPlus), `Actors Dataset` | Download as released. `class` 1 is illicit, 2 is licit. |
| `labels/recent_sample.csv` | blockchain.info unconfirmed-transaction pool | `bfx.bitcoin.sample_recent_addresses(10_000)`; the paper's sample was collected in early April 2025. |
| `transactions/` | blockchain.info `rawaddr` endpoint | `bfx.bitcoin.fetch_wallet`, see `notebooks/bitcoin/01_data_collection.ipynb`. |

Folder names under `transactions/` and `features/`: `chainabuse`, `elliptic_fraud`,
`elliptic_licit`, `unknown` (recent sample); the feature tables are
`chainabuse_metrics.csv`, `elliptic_fraud_metrics.csv`, `elliptic_licit_metrics.csv`
and `recent_sample_metrics.csv`.

## ERC-20

Each list is a CSV with `tokenname` and `contract_address`.

| File | Class | Source |
|---|---|---|
| `labels/tokenscout_scam.csv` | scam | Scam tokens linked to externally owned accounts in the TokenScout dataset (Wu et al., CCS 2024). |
| `labels/chainabuse_scam.csv` | scam | ERC-20 contracts named in ChainAbuse scam reports. |
| `labels/cmc_licit.csv` | legitimate | Ethereum tokens in the CoinMarketCap top-capitalisation listing. |
| `labels/tokenscout_licit.csv` | legitimate | Legitimate tokens of the TokenScout academic benchmark. |

Transfers are downloaded from the Etherscan v2 API with
`bfx.erc20.fetch_token_transfers` (first 500 transfers per contract). Set
`ETHERSCAN_API_KEY` in the environment; never write keys into notebooks or code.
Feature folders are named after the lists: `tokenscout_scam_tokens`,
`chainabuse_scam_tokens`, `cmc_licit_tokens`, `tokenscout_licit_tokens`.
