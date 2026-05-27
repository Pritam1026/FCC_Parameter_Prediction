# Configuration Loader for FCC Xgboost-LSTM-TA project
# Loads YAML config and exposes typed dataclasses

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# ------------------------------------------
# Dataclasses mirroring configs/config.yaml
# ------------------------------------------

@dataclass
class DataConfig:
    n_samples: int = 2600
    n_features: int = 52
    n_targets: int = 8
    top_k_features:int = 25
    train_ratio:float = 0.70
    seed:int = 42
    target_names:List[str] = field(default_factory=lambda: [
        "Gasoline", "Diesel", "Sdelurry", "Liquified_Gas",
        "Dry_Gas", "NOx", "CO2", "SO2"
    ])

@dataclass
class PreprocessingConfig:
    sliding_window_step:int = 5
    normalize:bool = True

@dataclass
class XGBoostConfig:
    n_estimators:int = 100
    max_depth:int = 6
    learning_rate:float = 0.8
    subsample:float = 0.8
    colsample_bytree:float = 0.8
    random_state:int = 42
    n_jobs:int = -1

@dataclass
class LSTMTAConfig:
    hidden_size:int = 128
    window_size:int = 12
    dropout_rate:float = 0.2
    batch_size:int = 128
    epochs: int = 100
    learning_rate: float = 0.001
    activation: str = "tanh"

@dataclass
class HyperparamSearchConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 128, 192, 256])
    window_sizes: List[int] = field(default_factory=lambda: [6, 12, 18, 24])

@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["MSE", "RMSE", "MAE", "MAPE", "R2"])

@dataclass
class PathConfig:
    data_dir:str = "outputs/data"
    model_dir:str = "outputs/models"
    results_dir:str = "outputs/results"
    log_dir:str = "logs"

@dataclass
class Config:
    data:DataConfig = field(default_factory=DataConfig)
    preprocessing:PreprocessingConfig = field(default_factory=PreprocessingConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lstm_ta: LSTMTAConfig = field(default_factory=LSTMTAConfig)
    hyperparameter_search: HyperparamSearchConfig = field(default_factory=HyperparamSearchConfig)
    evaluation:EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ----------------------------------------------------
# Loader
# ----------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file and return a configd
    dataclass

    Args:
        Config_path: Path to the YAML config file.
        Defaults to config/config.yaml
        relative to the project root.

    Returns:
        Populated config dataclass
    
    Raises:
        FileNotFoundError: if the config file doesnot exits
    """
    if config_path is None:
        # Resolve relative to this file's parent 
        # (src/utils) -> project root
        project_root = Path(__file__).resolve().parents[2]
        config_path = str(project_root/"Configs"/"config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        data = DataConfig(**raw.get('data',{})),
        preprocessing = PreprocessingConfig(**raw.get('preprocessing',{})),
        xgboost = XGBoostConfig(**raw.get('xgboost',{})),
        lstm_ta = LSTMTAConfig(**raw.get('lstm_ta',{})),
        hyperparameter_search = HyperparamSearchConfig(**raw.get('hyperparamter_search',{})),
        evaluation = EvaluationConfig(**raw.get('evaluation',{})),
        paths = PathConfig(**raw.get('paths', {}))
    )

    return cfg