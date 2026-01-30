from __future__ import annotations
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Literal
from dataclasses import dataclass, field
import json

import tensorflow as tf
import numpy as np

class Colours:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Standard colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if not attr.startswith('_') and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, "")
                
@dataclass
class LogLevel:
    name: str
    colour: str
    icon: str
    level: int
    
LOG_LEVELS = {
    "debug": LogLevel("DEBUG", Colours.DIM, "🔍", logging.DEBUG),
    "info": LogLevel("INFO", Colours.BLUE, "ℹ️ ", logging.INFO),
    "success": LogLevel("SUCCESS", Colours.BRIGHT_GREEN, "✓", logging.INFO + 1),
    "metric": LogLevel("METRIC", Colours.CYAN, "📊", logging.INFO + 2),
    "warning": LogLevel("WARNING", Colours.BRIGHT_YELLOW, "⚠️ ", logging.WARNING),
    "error": LogLevel("ERROR", Colours.BRIGHT_RED, "✗", logging.ERROR),
    "critical": LogLevel("CRITICAL", Colours.BG_RED + Colours.WHITE, "💀", logging.CRITICAL),
    "checkpoint": LogLevel("CHECKPOINT", Colours.BRIGHT_GREEN, "💾", logging.INFO + 3),
    "epoch": LogLevel("EPOCH", Colours.BRIGHT_MAGENTA, "🔄", logging.INFO + 4),
}

# Console Formatter
class ConsoleFormatter(logging.Formatter):
    def __init__(self, frmt: str | None = None, date_frmt: str | None = None, use_colours: bool = True):
        super().__init__(frmt,date_frmt)
        self.use_colours = use_colours

    def format(self, record: logging.LogRecord):
        # Getting the logging info
        level_name = record.levelname.lower()
        level_config = LOG_LEVELS.get(level_name, LOG_LEVELS['info'])

        # Checking if the colours are needed
        if self.use_colours:

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            coloured_time = f"{Colours.DIM}{timestamp}{Colours.RESET}"

            colourized_level = f"{level_config.colour}{level_config.icon}{level_config.name:10}{Colours.RESET}"

            match level_name:
                case "error" | "critical":
                    coloured_msg = f"{Colours.RED}{record.getMessage()}{Colours.RESET}"
                case "success":
                    coloured_msg = f"{Colours.GREEN}{record.getMessage()}{Colours.RESET}"
                case "checkpoint":
                    coloured_msg = f"{Colours.BRIGHT_GREEN}{record.getMessage()}{Colours.RESET}"
                case "warning":
                    coloured_msg = f"{Colours.YELLOW}{record.getMessage()}{Colours.RESET}"
                case "metric":
                    coloured_msg = f"{Colours.CYAN}{record.getMessage()}{Colours.RESET}"
                case "epoch":
                    coloured_msg = f"{Colours.MAGENTA}{record.getMessage()}{Colours.RESET}"
                case _:
                    coloured_msg = record.getMessage()

            return f"{coloured_time} | {colourized_level} | {coloured_msg}"
        else:
            super().format(record)
            
# File Formatter
class FileFormatter(logging.Formatter):
    def format(self,record: logging.LogRecord):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname

        # Including extra stuff I might want to add later
        extra = ""
        if hasattr(record,"step"):
            extra = extra + f" [step={record.step}"
        if hasattr(record,"epoch"):
            extra = extra + f" [epoch={record.epoch}"

        return f"{timestamp} | {level:10} | {record.getMessage()}{extra}"
    
# TensorBoard Writer
class TensorBoardWriter:
    def __init__(self, log_directory: Path):
        self.log_directory = log_directory
        self._writer = tf.summary.create_file_writer(str(log_directory))

    @property
    def writer(self):
        return self._writer

    def scalar(self, tag: str, value: float, step: int):
        with self.writer.as_default(step = step):
            tf.summary.scalar(tag, value)

    def scalars(self, main_tag: str, values: dict[str, float], step: int):
        with self.writer.as_default(step = step):
            for name, value in values.items():
                # Writing the scalars to the tensorboard
                tf.summary.scalar(f"{main_tag}/{name}", value)

    def image(self, tag: str, image: tf.Tensor | np.ndarray, step: int):
        with self._writer.as_default(step = step):
            # Writing the image to the tensorboard
            if len(image.shape) == 3:
                image = tf.expand_dims(image,axis = 0)

            tf.summary.image(tag, image)

    def histogram(self, tag: str, values: tf.Tensor | np.ndarray, step : int):
        with self._writer.as_default(step = step):
            tf.summary.histogram(tag, values)

    def text(self, tag: str, text: str, step: int):
        with self._writer.as_default(step = step):
            tf.summary.text(tag, text)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
        
# Main Logger

class Logger:
    def __init__(self, job_name: str, log_dir: str | Path = "logs", tensorboard: bool = True, console: bool = True, file: bool = True, level: str = "info", config: dict | None = None):
        
        self.job_name = job_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.job_dir = Path(log_dir) / f"{job_name}_{timestamp}"
        self.job_dir.mkdir(parents = True, exist_ok = True)
        
        self.tensorboard_dir = self.job_dir / "tensorboard"
        self.checkpoints_dir = self.job_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok = True)

        self._logger = logging.getLogger(f"training.{job_name}.{timestamp}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.propogate = False

        self._console_logging_enabled = console
        if console:
            console_logging_handler = logging.StreamHandler(sys.stdout)
            console_logging_handler.setLevel(LOG_LEVELS.get(level, LOG_LEVELS['info']).level)
            console_logging_handler.setFormatter(ConsoleFormatter())
            self._logger.addHandler(console_logging_handler)

        self._file_logging_enabled = file
        if file:
            log_file = self.job_dir / "training.log"
            file_logging_handler = logging.FileHandler(log_file, encoding = "utf-8")
            file_logging_handler.setLevel(logging.DEBUG)
            file_logging_handler.setFormatter(FileFormatter())
            self._logger.addHandler(file_logging_handler)

        self._tensorboard_writer: TensorBoardWriter | None = None
        if tensorboard:
            self.tensorboard_dir.mkdir(exist_ok = True)
            self._tensorboard_writer = TensorBoardWriter(self.tensorboard_dir)

        # Storing a metric history
        self._metric_history: list[dict] = []

        # Saving the config file snapshot for examination too
        if config:
            config_path = self.job_dir / "config.json"
            with open(config_path, "w") as file:
                json.dump(config, file, indent = 2, default = str)

        self.info(f"Logger Initialized: {self.job_dir}")

    def _log(self, level: str, message: str, **extra):
        level_config = LOG_LEVELS.get(level, LOG_LEVELS["info"])

        record = self._logger.makeRecord(name = self._logger.name, level = level_config.level, fn = "", lno = 0, msg = message, args = (), exc_info = None)

        record.level_name = level.upper()
        for key, value in extra.items():
            setattr(record, key, value)

        self._logger.handle(record)

    def debug(self, message: str, **extra):
        self._log("debug", message, **extra)

    def info(self, message: str, **extra):
        self._log("info", message, **extra)

    def success(self, message: str, **extra):
        self._log("success", message, **extra)

    def warning(self, message: str, **extra):
        self._log("warning", message, **extra)

    def critical(self, message: str, **extra):
        self._log("critical", message, **extra)

    def error(self, message: str, **extra):
        self._log("error", message, **extra)

    def checkpoint(self, message: str, path: str | Path | None = None, **extra):
        full_message = f"{message} -> {path}" if path else message
        self._log("checkpoint", message, **extra)

    def epoch(self, epoch: int, total: int | None = None, **extra):
        message = f"Epoch {epoch}/{total}" if total else f"Epoch {epoch}"
        self._log("epoch", message, epoch = epoch, **extra)

    def metric(self, message: str, **extra):
        self._log("metric", message, **extra)

    def log_scalar(self, tag: str, value: float, step: int):

        if self._tensorboard_writer:
            self._tensorboard_writer.scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict[str, float], step: int):
        if self._tensorboard_writer:
            self._tensorboard_writer.scalars(tag, values, step)

    def log_image(self, tag: str, image: tf.Tensor | np.ndarray, step: int):
        if self._tensorboard_writer:
            self._tensorboard_writer.image(tag, image, step)

    def log_histogram(self, tag: str, values: tf.Tensor | np.ndarray, step: int):
        if self._tensorboard_writer:
            self._tensorboard_writer.histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int):
        if self._tensorboard_writer:
            self._tensorboard_writer.text(tag, text, step)

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "", to_tensorboard: bool = True, to_console: bool = True):
        if prefix:
            prefixed = {f"{prefix}/{key}": value for key, value in metrics.items()}
        else:
            prefixed = metrics

        if to_tensorboard and self._tensorboard_writer:
            # Write to tensorboard
            for tag, value in prefixed.items():
                self._tensorboard_writer.scalar(tag, value, step)

        if to_console:
            metrics_message = " | ".join(f"{key}: {value:.4f}" for key, value in prefixed.items())
            self.metric(f"[Step {step}] {metrics_message}", step = step)

        # Adding to the metric history
        self._metric_history.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **prefixed
        })

    def log_training_step(self, step: int, loss: float, learning_rate: float, extra: dict[str,float] | None = None, log_every: int = 100):

        # Checking if the step needs to be logged
        if step % log_every != 0:
            return

        metrics = {"loss": loss, "lr": learning_rate}
        if extra:
            metrics.update(extra)

        self.log_metrics(metrics, step, prefix = "train", to_console = True)

    def log_validation(self, metrics: dict[str, float], step: int):

        self.log_metrics(metrics, step = step, prefix = "val", to_console = True)

        # Checking to highlight classic metrics
        for key in ["mAP@0.50", "mAP", "AP"]:
            if key in metrics:
                self.success(f"Validation {key}: {metrics[key]:.4f}")
                break

    def log_epoch_summary(self, epoch: int, train_metrics: dict[str,float], val_metrics: dict[str, float] | None = None):

        # Line divider
        self.info(f"{'-' * 50}")

        training_message = " | ".join(f"{key}: {value:.4f}" for key,value in train_metrics.items())
        self.info(f"Epoch {epoch} Train: {training_message}")

        if val_metrics:
            validation_message = " | ".join(f"{key}: {value:.4f}" for key,value in val_metrics.items())
            self.info(f"Epoch {epoch} Val: {validation_message}")

        self.info(f"{'-' * 50}")

    def get_checkpoint_path(self, filename: str):
        return self.checkpoints_dir / filename

    def save_metric_history(self):
        path = self.job_dir / "metric_history.json"
        with open(path, "w") as file:
            json.dump(self._metric_history, file, indent = 2)

    def flush(self):
        # Flushing each handler
        for handler in self._logger.handlers:
            handler.flush()

        if self._tensorboard_writer:
            self._tensorboard_writer.flush()

    def close(self):

        # Wrapping up everything
        self.save_metric_history()

        if self._tensorboard_writer:
            self._tensorboard_writer.close()

        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.error(f"Exception: {exc_type.__name__}: {exc_val}")

        self.close()
        
def build_logger_from_config(config: dict, job_name: str | None = None):
    logging_config = config.get('logging', {})

    return Logger(job_name = job_name, log_dir = logging_config.get('log_dir', "logs"), tensorboard = logging_config.get('tensorboard', True), console = logging_config.get('console', True), file = logging_config.get('file', True), level = logging_config.get('level', "info"))

