from __future__ import annotations
import logging
import sys
import subprocess
import socket
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Literal
from dataclasses import dataclass, field
from mobilenetv2ssd.core.fingerprint import Fingerprint
from training.ema import EMA
import json

import tensorflow as tf
import numpy as np

# TODO: Add Progress bars for training and validation loops, maybe using tqdm or a custom implementation that works well with the logging system
# TODO: Add Timezone standatdization for timestamps, maybe using UTC or allowing the user to specify a timezone in the config

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
            
        self.flush()

    def scalars(self, main_tag: str, values: dict[str, float], step: int):
        with self.writer.as_default(step = step):
            for name, value in values.items():
                # Writing the scalars to the tensorboard
                tf.summary.scalar(f"{main_tag}/{name}", value)
                
        self.flush()

    def image(self, tag: str, image: tf.Tensor | np.ndarray, step: int):
        with self.writer.as_default(step = step):
            # Writing the image to the tensorboard
            if len(image.shape) == 3:
                image = tf.expand_dims(image,axis = 0)

            tf.summary.image(tag, image)
            
        self.flush()
            

    def histogram(self, tag: str, values: tf.Tensor | np.ndarray, step : int):
        with self.writer.as_default(step = step):
            tf.summary.histogram(tag, values)
        
        self.flush()

    def text(self, tag: str, text: str, step: int):
        with self.writer.as_default(step = step):
            tf.summary.text(tag, text)

        self.flush()
        
    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
        
# Main Logger

class Logger:
    def __init__(self, job_name: str, log_dir: str | Path = "logs", tensorboard: bool = True, console: bool = True, file: bool = True, level: str = "info", config: dict | None = None):
        
        self.job_name = job_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.job_dir = Path(log_dir) / timestamp
        self.job_dir.mkdir(parents = True, exist_ok = True)
        
        self.tensorboard_dir = self.job_dir / "tensorboard"
        self.checkpoints_dir = self.job_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok = True)

        self._logger = logging.getLogger(f"training.{job_name}.{timestamp}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.propagate = False

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
            
            try:
                self.start_tensorboard()
            except Exception as e:
                self.warning(f"Failed to start TensorBoard server: {e}")


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
        self._log("checkpoint", full_message, **extra)

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

         # Stop TensorBoard if running
        if hasattr(self, '_tensorboard_process') and self._tensorboard_process is not None:
            if self._tensorboard_process.poll() is None:
                self.info("Stopping TensorBoard server...")
                self._tensorboard_process.terminate()
                try:
                    self._tensorboard_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._tensorboard_process.kill()
                    self._tensorboard_process.wait()

        if self._tensorboard_writer:
            self._tensorboard_writer.close()

        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)
            
    def start_tensorboard(self, port: int = 6006, host: str = "0.0.0.0", log_dir: str | Path | None = None):
        
        # Use provided log_dir or default to tensorboard_dir
        target_dir = Path(log_dir).resolve() if log_dir else self.tensorboard_dir
        
        if not target_dir.exists():
            raise ValueError(f"Log directory does not exist: {target_dir}")
        
        # Check if our stored process is still running
        if hasattr(self, '_tensorboard_process') and self._tensorboard_process is not None:
            if self._tensorboard_process.poll() is None:
                self.info(f"TensorBoard is already running on port {port}")
                return self._tensorboard_process
            else:
                # Process died, clean up
                self._tensorboard_process = None
        
        # Check if any process is using the port (could be from a previous run)
        def is_port_in_use(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('0.0.0.0', port))
                    return False
                except OSError:
                    return True
        
        if is_port_in_use(port):
            self.warning(f"Port {port} is already in use - TensorBoard may already be running")
            self.info(f"Try accessing: http://localhost:{port}")
            self.info(f"To use a different port, call: logger.start_tensorboard(port=XXXX)")
            return None
        
        if not target_dir.exists():
            raise ValueError(f"Log directory does not exist: {target_dir}")
        
        # Check if TensorBoard is already running
        if hasattr(self, '_tensorboard_process') and self._tensorboard_process is not None:
            if self._tensorboard_process.poll() is None:
                self.warning("TensorBoard is already running")
                return self._tensorboard_process
        
        # Start TensorBoard
        cmd = [
            "tensorboard",
            "--logdir", str(target_dir),
            "--port", str(port),
            "--host", host
        ]
        
        self.info(f"Starting TensorBoard on port {port}...")
        self.info(f"Log directory: {target_dir}")
        
        self._tensorboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give TensorBoard time to start
        time.sleep(3)
        
        # Check if process started successfully
        if self._tensorboard_process.poll() is not None:
            _, stderr = self._tensorboard_process.communicate()
            self.error(f"TensorBoard failed to start: {stderr}")
            raise RuntimeError(f"TensorBoard failed to start: {stderr}")
        
        # Get network information for access instructions
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "YOUR_IP_ADDRESS"
        
        # Log access information
        self.success("TensorBoard is running!")
        self.info("-" * 60)
        self.info("Access URLs:")
        self.info(f"  Local:        http://localhost:{port}")
        self.info(f"  Network:      http://{local_ip}:{port}")
        self.info(f"  Remote/Cloud: http://YOUR_SERVER_IP:{port}")
        self.info("-" * 60)
        self.info("For AWS/Cloud instances:")
        self.info(f"  1. Ensure security group allows inbound traffic on port {port}")
        self.info(f"  2. Use: http://<your-instance-public-ip>:{port}")
        self.info("")
        self.info("For SSH port forwarding (more secure, no open ports needed):")
        self.info(f"  ssh -L {port}:localhost:{port} user@remote-server")
        self.info(f"  Then access: http://localhost:{port}")
        self.info("-" * 60)
        
        return self._tensorboard_process
    
    def save_model_weights(self, model: tf.keras.Model, training_result: dict, config: dict[str, Any], fingerprint: Fingerprint | None = None, ema: EMA = None):
        # Creating the weights directory
        
        weights_dir = self.job_dir / "weights"
        weights_dir.mkdir(exist_ok= True)
        
        # Save EMA weights if they exist
        if ema is not None and ema.enabled and ema.eval_use_ema:
            ema.apply_to(model= model)
            self.checkpoint(f"Applied EMA shadow weights to model.")
            
        weights_path = weights_dir / "final_weights.weights.h5"
        model.save_weights(str(weights_path))
        self.checkpoint(f"Saved model final weights to {weights_path}")
        
        summary = {
            "job_name": self.job_name,
            "best_metric": training_result.get("best_metric"),
            "primary_metric": training_result.get("primary_metric"),
            "global_step": training_result.get("global_step"),
            "weights_path": str(weights_path),
            "fingerprint": str(fingerprint.short) if fingerprint else None,
            "config": config,
        }
        
        summary_path = self.job_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
            
        self.success(f"Saved training summary to {summary_path}")
        
    def stop_tensorboard(self):

        if hasattr(self, '_tensorboard_process') and self._tensorboard_process is not None:
            if self._tensorboard_process.poll() is None:
                self.info("Stopping TensorBoard...")
                self._tensorboard_process.terminate()
                try:
                    self._tensorboard_process.wait(timeout=5)
                    self.success("TensorBoard stopped")
                except subprocess.TimeoutExpired:
                    self._tensorboard_process.kill()
                    self._tensorboard_process.wait()
                    self.warning("TensorBoard forcefully killed")
            else:
                self.info("TensorBoard is not running")
            self._tensorboard_process = None
        else:
            self.info("No TensorBoard process found")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.error(f"Exception: {exc_type.__name__}: {exc_val}")

        self.close()
        
def build_logger_from_config(config: dict, fingerprint: Fingerprint = None):
    
    # Check if the run directory is specified in the config, if not use a default one
    experiment_name = config.get("experiment", {}).get("id", "default_experiment")
    fingerprint_str = str(fingerprint.short) if fingerprint else "no_fingerprint"
    name_format = config.get("run", {}).get('name_format', "{experiment_id}_{fingerprint}")
    
    name_format = name_format.replace("{experiment_id}", experiment_name)
    name_format = name_format.replace("{fingerprint}", fingerprint_str)
    
    job_name = name_format
    run_dir = config.get("run", {}).get("root", "runs")
    log_dir = config.get("run", {}).get("logs", "logs")
    
    run_dir = Path(run_dir)
    log_dir = run_dir / job_name / log_dir
    
    tensorboard = config.get("logging", {}).get("tensorboard", {}).get("enabled", True)
    file = config.get("logging", {}).get("file", {}).get("enabled", True)
    console = config.get("logging", {}).get("console", {}).get("enabled", True)
    level = config.get("logging", {}).get("level", "info")
    
    return Logger(job_name = job_name, log_dir = log_dir, tensorboard = tensorboard, console = console, file = file, level = level)