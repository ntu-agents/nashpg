"""
Logging utilities for training
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf


class BaseLogger(ABC):
    """Base class for all loggers"""
    
    @abstractmethod
    def log_config(self, config: DictConfig) -> None:
        """Log the configuration"""
        pass
    
    @abstractmethod
    def log_train_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics"""
        pass
    
    @abstractmethod
    def log_rollout_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log rollout metrics"""
        pass
    
    @abstractmethod
    def log_eval_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log evaluation metrics"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the logger"""
        pass


class TensorBoardLogger(BaseLogger):
    """TensorBoard logger"""
    
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
    
    def log_config(self, config: DictConfig) -> None:
        """Log config as text to TensorBoard"""
        config_text = OmegaConf.to_yaml(config)
        self.writer.add_text("config", config_text, 0)
    
    def log_train_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics"""
        for key, val in metrics.items():
            self.writer.add_scalar(f"train/{key}", val.item(), step)
    
    def log_rollout_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log rollout metrics"""
        for key, val in metrics.items():
            self.writer.add_scalar(f"rollout/{key}", val.item(), step)
    
    def log_eval_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log evaluation metrics"""
        for key, val in metrics.items():
            self.writer.add_scalar(f"eval/{key}", val.item(), step)
    
    def close(self) -> None:
        """Close TensorBoard writer"""
        self.writer.close()


class JSONLogger(BaseLogger):
    """JSON file logger"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log structure
        self.log_data = {
            "config": None,
            "train": [],
            "eval": [],
            "rollout": []
        }
    
    def log_config(self, config: DictConfig) -> None:
        """Log the configuration"""
        self.log_data["config"] = OmegaConf.to_container(config, resolve=True)
        self._write_to_file()
    
    def log_train_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics"""
        log_entry = {"step": step}
        for key, val in metrics.items():
            log_entry[key] = val.item()
        self.log_data["train"].append(log_entry)
        # self._write_to_file()
    
    def log_rollout_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log rollout metrics"""
        log_entry = {"step": step}
        for key, val in metrics.items():
            log_entry[key] = val.item()
        self.log_data["rollout"].append(log_entry)
        # self._write_to_file()
    
    def log_eval_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log evaluation metrics"""
        log_entry = {"step": step}
        for key, val in metrics.items():
            log_entry[key] = val.item()
        self.log_data["eval"].append(log_entry)
        # self._write_to_file()
    
    def _write_to_file(self) -> None:
        """Write current log data to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def close(self) -> None:
        """Close JSON logger (final write)"""
        self._write_to_file()


class MultiLogger(BaseLogger):
    """Logger that combines multiple loggers"""
    
    def __init__(self, loggers: List[BaseLogger]):
        self.loggers = loggers
    
    def log_config(self, config: DictConfig) -> None:
        """Log config to all loggers"""
        for logger in self.loggers:
            logger.log_config(config)
    
    def log_train_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics to all loggers"""
        for logger in self.loggers:
            logger.log_train_metrics(metrics, step)
    
    def log_rollout_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log rollout metrics to all loggers"""
        for logger in self.loggers:
            logger.log_rollout_metrics(metrics, step)
    
    def log_eval_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log evaluation metrics to all loggers"""
        for logger in self.loggers:
            logger.log_eval_metrics(metrics, step)
    
    def close(self) -> None:
        """Close all loggers"""
        for logger in self.loggers:
            logger.close()


def create_logger(config: DictConfig) -> BaseLogger:
    """Create logger based on config"""
    loggers = []
    
    # Create TensorBoard logger if enabled
    if config.logging.get("tensorboard", {}).get("enabled", True):
        tb_dir = config.logging.get("tensorboard", {}).get("log_dir", f"runs/{config.run_name}")
        loggers.append(TensorBoardLogger(tb_dir))
    
    # Create JSON logger if enabled
    if config.logging.get("json", {}).get("enabled", False):
        json_file = config.logging.get("json", {}).get("log_file", f"logs/{config.run_name}.json")
        loggers.append(JSONLogger(json_file))
    
    if len(loggers) == 0:
        raise ValueError("No loggers enabled in config")
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return MultiLogger(loggers)