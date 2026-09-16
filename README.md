# Alpha ML: Cross-Sectional Equity Ranking

This project develops an end-to-end machine learning pipeline for cross-sectional equity ranking. Using historical price, volume, volatility, trend, liquidity, and rank-based features, the models estimate whether each stock's future relative return will fall in the daily bottom 20%, middle 60%, or top 20%.

The primary experiment uses a 252-trading-day lookback and predicts relative-return classes over 1, 3, 5, 10, 20, and 60 trading-day horizons.

## Project workflow

1. Collect and clean daily U.S. equity market data.
2. Construct return, volatility, trend, liquidity, and cross-sectional rank features.
3. Create three-class relative-return targets: bottom 20%, middle 60%, and top 20%.
4. Create targets for 1, 3, 5, 10, 20, and 60 trading-day horizons.
5. Prevent look-forward leakage by requiring each target end date to remain within its train or validation split.
6. Compare LightGBM with FCNN, CNN, GRU, Transformer, CNN+GRU, and CNN+Transformer models.
7. Select the best model for each horizon using validation mean daily cross-sectional macro AUC.
8. Refit each selected model on the combined training and validation data.
9. Generate out-of-sample class probabilities for the test period.

The neural networks first encode each stock's 252-day historical sequence. They then apply cross-sectional attention across the stocks available on the same date, allowing each prediction to incorporate both the stock's own history and information from the broader universe.

## Test results

The following table reports the out-of-sample performance of the model selected for each prediction horizon.

| Horizon | Selected model | Accuracy | Balanced accuracy | Macro F1 | Mean daily macro AUC | Mean daily extreme AUC | Log loss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 day | CNN+Transformer | 0.6056 | 0.3505 | 0.2917 | 0.6327 | 0.6196 | 0.9088 |
| 3 days | CNN+Transformer | 0.6056 | 0.3586 | 0.3084 | 0.6357 | 0.6221 | 0.9075 |
| 5 days | CNN+Transformer | 0.6057 | 0.3570 | 0.3094 | 0.6388 | 0.6241 | 0.9057 |
| 10 days | CNN+Transformer | 0.6050 | 0.3500 | 0.2919 | 0.6357 | 0.6211 | 0.9080 |
| 20 days | CNN+Transformer | 0.6046 | 0.3481 | 0.2885 | 0.6353 | 0.6213 | 0.9087 |
| 60 days | CNN+GRU | 0.5976 | 0.3726 | 0.3465 | 0.6264 | 0.6082 | 0.9205 |

Mean daily macro AUC is calculated by computing the one-vs-rest AUC for the bottom, middle, and top classes within each date, averaging the three class AUCs, and then averaging across test dates.

Mean daily extreme AUC is the average of the daily bottom-class and top-class AUCs. It focuses on the two classes most relevant to identifying relatively weak and strong stocks.

## Interpretation

The test mean daily macro AUC ranges from approximately 0.626 to 0.639, above the random-ranking benchmark of 0.5. This indicates that the models contain measurable out-of-sample information about the relative ordering of stocks.

These results suggest that the model probabilities contain useful cross-sectional ranking information. However, the signal is modest, and the classification metrics do not directly determine economically meaningful portfolio weights.

### Why accuracy is not sufficient

The target distribution is approximately:

- bottom class: 20%;
- middle class: 60%;
- top class: 20%.

A naive classifier that always predicts the middle class can therefore achieve approximately 60% accuracy without successfully identifying strong or weak stocks.

The model accuracy values are also close to 60%, so accuracy alone does not demonstrate useful extreme-stock selection. This is why balanced accuracy, macro F1, and class-specific AUC are also reported.

The balanced accuracy values range from approximately 0.35 to 0.37, only moderately above the three-class random benchmark of approximately 0.333. This shows that distinguishing all three classes remains difficult, particularly because the middle class is much larger than either extreme class.

### Why the probabilities do not directly produce weights

The models predict relative-return classes rather than future return magnitudes. They estimate:

- the probability of finishing in the bottom 20%;
- the probability of finishing in the middle 60%;
- the probability of finishing in the top 20%.

They do not directly estimate:

- the expected return of each stock;
- the size of the potential gain or loss;
- the volatility of each position;
- correlations between stock returns;
- prediction uncertainty beyond the class probabilities;
- the appropriate amount of capital to allocate to each stock.

A stock with a high probability of finishing in the top 20% is not necessarily expected to outperform by a large amount. Similarly, two stocks with similar class probabilities may have very different return distributions and risk characteristics.

Therefore, the test AUC results provide evidence of a ranking signal, but they are not sufficient by themselves to determine an optimal mapping from probabilities to portfolio weights.

## Repository structure

text
alpha_ml/
├── notebooks/
│   ├── 00_*.ipynb              # Data collection and project setup
│   ├── 01_*.ipynb              # Feature engineering and dataset construction
│   ├── 02_*.ipynb              # Multi-horizon model training and evaluation
│   └── experiments/            # Additional exploratory model notebooks
├── figures/                    # Selected figures
├── results/                    # Selected summary metrics
├── requirements.txt
├── LICENSE
└── README.md

Large datasets, fitted models, checkpoints, prediction files, and other reproducible intermediate outputs are excluded from GitHub.

## Running the project

1. Install the packages listed in requirements.txt.
2. Run notebook 00 to collect and organize the market data.
3. Run notebook 01 to construct the features, targets, and train-validation-test splits.
4. Run notebook 02 with a GPU runtime to train the candidate models, select the best model for each horizon, refit the selected models, and evaluate them on the test period.

## Limitations and future work

- The available constituent list may introduce survivorship bias.
- The three-class target does not measure future return magnitude.
- The middle class accounts for approximately 60% of observations, creating an imbalanced classification problem.
- Accuracy is dominated by the middle class and is not an adequate standalone model-selection metric.
- AUC measures ranking quality but does not directly optimize expected return, risk, or capital allocation.
- A single train-validation-test split provides less evidence of temporal stability than walk-forward evaluation.

Future work will investigate alternative prediction targets, class-balanced and ranking-based losses, expected-return estimation, probability calibration, uncertainty estimation, and objectives more closely aligned with portfolio-weight construction.

This repository is a research project and does not constitute investment advice.
