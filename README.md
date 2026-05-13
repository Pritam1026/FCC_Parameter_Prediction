# XGBOOST-LSTM-TA: FCC Process Prediction (TensorFlow)

Implementation of the prediction framework from:

> **"Efficient prediction framework for large-scale nonlinear petrochemical process based on feature selection and temporal-attention LSTM: Applied to fluid catalytic cracking"**  
> Jian Long, Long Ye, Haifei Peng, Zhou Tian  
> *Chemical Engineering Science 301 (2025) 120733*

---

## Repository Status

**No official code was found** for this paper. The data is explicitly marked confidential
(`"The data that has been used is confidential"` — Section: Data Availability).

This implementation uses **synthetic data** that mimics the paper's described statistics:
- 2600 samples, 52 features, 8 targets (5 yields + 3 flue gas)
- Temporal correlations with realistic time delays (diesel ~10min, gasoline ~50min)
- AR(1) process-driven features with operational mode shifts

---

## Folder Structure

```
fcc_xgboost_lstm_ta/
├── configs/
│   └── config.yaml              # All hyperparameters (mirrors paper)
├── src/
│   ├── data/
│   │   ├── data_generator.py    # Synthetic FCC data generator
│   │   └── preprocessor.py      # SVM denoising + sliding window + normalization
│   ├── models/
│   │   ├── feature_selector.py  # XGBoost-based feature selection (Sec 3.2)
│   │   └── lstm_ta_model.py     # Temporal-Attention LSTM model (Sec 3.3-3.5)
│   ├── training/
│   │   └── trainer.py           # Training loop + grid hyperparameter search
│   ├── evaluation/
│   │   └── metrics.py           # MSE, RMSE, MAE, MAPE, R² + plotting
│   └── utils/
│       ├── logger.py            # Logging utility
│       └── config_loader.py     # YAML config → typed dataclasses
├── outputs/
│   ├── data/                    # Generated CSV data
│   ├── models/                  # Saved model checkpoints (.keras)
│   └── results/                 # Metrics CSV + prediction plots
├── logs/                        # Log files
├── main.py                      # Full pipeline entry point
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (generates data, trains, evaluates)
python main.py

# 3. Run for a single target only
python main.py --target Gasoline

# 4. Custom config
python main.py --config configs/config.yaml
```

---

## Model Architecture

```
Input (batch, T=12, m=25)        ← T=window, m=top-25 features
        ↓
  Encoder LSTM                   ← Returns ALL hidden states h₁..hT
  (hidden=128, return_sequences)
        ↓
  Temporal Attention Layer        ← Computes β weights over time steps
  l_t^i = v_d^T tanh(W_d·d + U_d·h_i)   (Eq. 12)
  β_t^i = softmax(l_t^i)                  (Eq. 13)
  c_t   = Σ β_t^i · h_i                   (Eq. 14)
        ↓
  Decoder LSTM (LSTMCell)
        ↓
  Dropout(0.2) → Dense(n_targets)
        ↓
  Output prediction ŷ_T                    (Eq. 17)
```

---

## Key Hyperparameters (from paper)

| Parameter      | Value(s)           | Source |
|----------------|--------------------|--------|
| hidden_size    | 64, **128**, 192, 256 | Sec 4.1.4, Fig 7(a) |
| window_size    | 6, **12**, 18, 24  | Sec 4.1.4, Fig 7(b) |
| batch_size     | 128                | Sec 4.1.4 |
| dropout        | 0.2                | Sec 4.1.4 |
| optimizer      | Adam               | Sec 4.1.4 |
| loss           | MSE                | Sec 3.4  |
| activation     | Tanh               | Sec 4.1.4 |
| top-K features | 25 (from 52)       | Sec 4.1.3 |
| train/test     | 70% / 30%          | Sec 4.1.2 |

---

## Targets and Expected Performance (from paper Table 1, FCC test set)

| Target     | MAE      | R²     |
|------------|----------|--------|
| Gasoline   | ~0.055   | ~0.979 |
| Diesel     | ~0.018   | ~0.963 |
| Slurry     | ~0.095   | ~0.987 |
| Liquefied Gas | ~0.038 | ~0.984 |
| Dry Gas    | ~0.016   | ~0.975 |
| NOx        | ~1.082   | ~0.978 |
| CO2        | ~0.677   | ~0.980 |
| SO2        | ~0.884   | ~0.977 |

*Note: These are paper-reported values on real industrial data. Synthetic data results will differ.*

---

## PyTorch Version

A PyTorch implementation will be added upon request.
