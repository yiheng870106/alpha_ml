# Alpha ML: Cross-Sectional Equity Ranking

This project develops an end-to-end machine learning pipeline for cross-sectional equity ranking. The models use historical price, volume, volatility, moving-average, and rank-based features to estimate whether each stock's future relative return will fall in the daily bottom 20%, middle 60%, or top 20%.

The primary experiment uses a 252-trading-day lookback. Alternative 60-day-lookback and directional-target versions are retained in `notebooks/experiments/`.

## Project workflow

1. Collect and clean daily equity market data.
2. Construct return, volatility, trend, liquidity, and cross-sectional rank features.
3. Create multi-horizon targets for 1, 3, 5, 10, 20, and 60 trading days.
4. Prevent look-forward leakage by requiring each target end date to remain inside its train or validation split.
5. Compare LightGBM with FCNN, CNN, GRU, Transformer, CNN+GRU, and CNN+Transformer models.
6. Select the best model for each horizon using validation mean daily cross-sectional AUC.
7. Refit the selected models on train plus validation data and generate out-of-sample probabilities.
8. Test market-neutral portfolio rules with overlapping holding periods and transaction costs.

The neural networks first encode each stock's historical sequence and then use cross-sectional attention so that predictions for one stock can incorporate information from the other stocks available on the same date.

## Main modeling result

The 252-day-lookback experiment produced out-of-sample mean daily macro AUC values of approximately 0.63 across the tested horizons. This indicates measurable cross-sectional ranking information, but predictive accuracy alone did not translate into a robust tradable strategy.

## Why the portfolio was not economically effective

The strongest directional market-neutral test retained a modest gross signal, but most of the edge disappeared after transaction costs.

| Test metric | Result |
|---|---:|
| Gross Sharpe ratio | 0.491 |
| Net Sharpe ratio at 10 bps | 0.017 |
| Net annualized return | -0.08% |
| Maximum drawdown | -7.54% |
| Mean daily turnover | 5.59% |
| Estimated break-even cost | 10.35 bps |

The main reasons are:

- classification AUC measures ranking quality, not return magnitude or portfolio Sharpe ratio;
- the predictive edge was small relative to turnover and assumed trading costs;
- dollar neutrality did not eliminate beta, sector, or other factor exposures;
- performance was unstable across market regimes and was weaker on the short side;
- overlapping targets and estimation error reduced effective out-of-sample robustness.

The negative portfolio result is kept intentionally. It demonstrates the difference between statistically detectable prediction ability and economically useful performance after implementation costs.

## Repository structure

```text
alpha_ml/
├── notebooks/                  # Main data, feature, and modeling notebooks
│   └── experiments/            # 60-day and directional research variants
├── figures/                    # Selected charts for documentation
├── results/                    # Small summary tables, when included
├── requirements.txt
├── LICENSE
└── README.md
```

Large datasets, fitted models, checkpoints, prediction files, and portfolio-level intermediate outputs are excluded from GitHub.

## Running the project

1. Install the packages in `requirements.txt`.
2. Run the data collection and feature engineering notebooks in order.
3. Run the primary 252-day-lookback model notebook with a GPU runtime.
4. Use the notebooks in `notebooks/experiments/` for alternative lookbacks and directional portfolio research.

Full neural-network training is computationally expensive. A reduced development configuration should be used first to verify the pipeline before running all horizons and model architectures.

## Limitations

- The available constituent list can introduce survivorship bias.
- The portfolio tests do not yet fully neutralize beta, sector, and common risk-factor exposures.
- Model selection is based on predictive metrics rather than a differentiable portfolio objective.
- A single train-validation-test split does not measure stability as thoroughly as walk-forward evaluation.

Future work will explore alternative targets and losses, factor neutralization, turnover-aware objectives, and walk-forward testing.

This repository is a research project and does not constitute investment advice.
