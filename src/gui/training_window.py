"""GUI window for energy zone classification model training."""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QPushButton, QTextEdit,
                             QProgressBar, QCheckBox, QSpinBox, QFileDialog,
                             QComboBox, QTableWidget, QTableWidgetItem, QTabWidget,
                             QWidget, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor
from pathlib import Path
import json

from ..training import ZoneTrainer
from ..utils import get_logger

logger = get_logger(__name__)


class TrainingThread(QThread):
    """Background thread for model training."""

    # Signals
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    log_message = pyqtSignal(str, str)  # level, message
    training_completed = pyqtSignal(dict)  # results
    training_failed = pyqtSignal(str)  # error message

    def __init__(self, data_path: str, algorithms: list, save_dir: str, kwargs: dict):
        super().__init__()
        self.data_path = data_path
        self.algorithms = algorithms
        self.save_dir = save_dir
        self.kwargs = kwargs
        self._is_running = True
        self.trainer = None  # Will be set in run()

    def run(self):
        """Run training in background."""
        try:
            # use_fast_mode=True: быстрое извлечение признаков (10x быстрее, хорошая точность)
            # use_embeddings=False: без wav2vec2 (еще быстрее)
            self.trainer = ZoneTrainer(
                test_data_path=self.data_path,
                use_fast_mode=True,  # БЫСТРО: ~3s/track вместо ~30s/track
                use_embeddings=False
            )

            results = self.trainer.run_full_training_pipeline(
                algorithms=self.algorithms,
                save_dir=self.save_dir,
                progress_callback=self._progress_callback,
                log_callback=self._log_callback,
                **self.kwargs
            )

            self.training_completed.emit(results)

        except Exception as e:
            logger.exception("Training failed")
            self.training_failed.emit(str(e))

    def _progress_callback(self, current: int, total: int, message: str):
        """Progress update callback."""
        if self._is_running:
            self.progress_updated.emit(current, total, message)

    def _log_callback(self, level: str, message: str):
        """Log message callback."""
        if self._is_running:
            self.log_message.emit(level, message)

    def stop(self):
        """Stop training gracefully."""
        self._is_running = False
        if self.trainer:
            self.trainer.request_stop()
            self.log_message.emit("INFO", "Requesting graceful stop... Progress will be saved at next checkpoint.")
        # Don't terminate() immediately - let trainer save checkpoint first
        # The thread will exit naturally when trainer stops


class TrainingWindow(QDialog):
    """GUI window for training energy zone classification models."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Zone Classifier")
        self.resize(900, 700)

        self.training_thread = None
        self.results = None

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Data selection
        data_group = self._create_data_selection_group()
        layout.addWidget(data_group)

        # Algorithm selection
        algo_group = self._create_algorithm_group()
        layout.addWidget(algo_group)

        # Settings
        settings_group = self._create_settings_group()
        layout.addWidget(settings_group)

        # Control buttons
        control_layout = self._create_control_buttons()
        layout.addLayout(control_layout)

        # Progress
        progress_group = self._create_progress_group()
        layout.addWidget(progress_group)

        # Tabs for logs and results
        tabs = self._create_tabs()
        layout.addWidget(tabs)

        self.setLayout(layout)

    def _create_data_selection_group(self) -> QGroupBox:
        """Create data file selection group."""
        group = QGroupBox("Training Data")
        layout = QHBoxLayout()

        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Select training data file (TSV with Zone labels: yellow/green/purple)...")
        layout.addWidget(self.data_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_data_file)
        layout.addWidget(browse_btn)

        group.setLayout(layout)
        return group

    def _create_algorithm_group(self) -> QGroupBox:
        """Create algorithm selection group."""
        group = QGroupBox("Algorithms")
        layout = QHBoxLayout()

        self.xgb_check = QCheckBox("XGBoost")
        self.xgb_check.setChecked(True)
        layout.addWidget(self.xgb_check)

        self.nn_check = QCheckBox("Neural Network")
        self.nn_check.setChecked(True)
        layout.addWidget(self.nn_check)

        self.ensemble_check = QCheckBox("Ensemble")
        self.ensemble_check.setChecked(True)
        layout.addWidget(self.ensemble_check)

        layout.addStretch()

        group.setLayout(layout)
        return group

    def _create_settings_group(self) -> QGroupBox:
        """Create training settings group."""
        group = QGroupBox("Settings")
        layout = QHBoxLayout()

        # Grid search
        self.grid_search_check = QCheckBox("Grid Search")
        self.grid_search_check.setToolTip("Perform hyperparameter grid search (slower but better)")
        layout.addWidget(self.grid_search_check)

        # Epochs (for NN)
        layout.addWidget(QLabel("NN Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(50, 500)
        self.epochs_spin.setValue(200)
        layout.addWidget(self.epochs_spin)

        # Log level
        layout.addWidget(QLabel("Log Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.setToolTip("Minimum log level to display")
        layout.addWidget(self.log_level_combo)

        # Save directory
        layout.addWidget(QLabel("Save to:"))
        self.save_dir_edit = QLineEdit("models/zone_classifiers")
        layout.addWidget(self.save_dir_edit)

        group.setLayout(layout)
        return group

    def _create_control_buttons(self) -> QHBoxLayout:
        """Create control buttons."""
        layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self._start_training)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        return layout

    def _create_progress_group(self) -> QGroupBox:
        """Create progress indicators."""
        group = QGroupBox("Progress")
        layout = QVBoxLayout()

        # Overall progress
        self.overall_progress = QProgressBar()
        self.overall_progress.setValue(0)
        layout.addWidget(QLabel("Overall Progress:"))
        layout.addWidget(self.overall_progress)

        # Current task
        self.current_task_label = QLabel("Ready")
        layout.addWidget(self.current_task_label)

        group.setLayout(layout)
        return group

    def _create_tabs(self) -> QTabWidget:
        """Create tabs for logs and results."""
        tabs = QTabWidget()

        # Logs tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        tabs.addTab(self.log_text, "Logs")

        # Results tab
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Model", "Overall Accuracy", "Yellow Acc.", "Green Acc.", "Purple Acc.", "Model Path"
        ])
        tabs.addTab(self.results_table, "Results")

        return tabs

    def _browse_data_file(self):
        """Browse for training data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Data File",
            "",
            "TSV Files (*.txt *.tsv);;All Files (*)"
        )
        if file_path:
            self.data_path_edit.setText(file_path)

    def _start_training(self):
        """Start training process."""
        # Validate inputs
        data_path = self.data_path_edit.text().strip()
        if not data_path or not Path(data_path).exists():
            QMessageBox.warning(self, "Error", "Please select a valid training data file")
            return

        # Get selected algorithms
        algorithms = []
        if self.xgb_check.isChecked():
            algorithms.append('xgboost')
        if self.nn_check.isChecked():
            algorithms.append('neural_network')
        if self.ensemble_check.isChecked():
            algorithms.append('ensemble')

        if not algorithms:
            QMessageBox.warning(self, "Error", "Please select at least one algorithm")
            return

        # Prepare kwargs
        kwargs = {
            'grid_search': self.grid_search_check.isChecked(),
            'epochs': self.epochs_spin.value()
        }

        save_dir = self.save_dir_edit.text().strip() or "models/zone_classifiers"

        # Clear previous logs and results
        self.log_text.clear()
        self.results_table.setRowCount(0)
        self.overall_progress.setValue(0)

        # Start training thread
        self._log("INFO", "Starting training process...")
        self.training_thread = TrainingThread(data_path, algorithms, save_dir, kwargs)
        self.training_thread.progress_updated.connect(self._on_progress_updated)
        self.training_thread.log_message.connect(self._log)
        self.training_thread.training_completed.connect(self._on_training_completed)
        self.training_thread.training_failed.connect(self._on_training_failed)
        self.training_thread.start()

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop_training(self):
        """Stop training process."""
        if self.training_thread:
            self._log("WARNING", "Stopping training...")
            self.training_thread.stop()
            self.training_thread.wait()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_progress_updated(self, current: int, total: int, message: str):
        """Handle progress updates."""
        if total > 0:
            progress_pct = int((current / total) * 100)
            self.overall_progress.setValue(progress_pct)

        self.current_task_label.setText(f"[{current}/{total}] {message}")

    def _log(self, level: str, message: str):
        """Add log message."""
        # Filter by log level
        log_levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "SUCCESS": 1}
        current_level = self.log_level_combo.currentText()
        min_level = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}[current_level]

        # Skip if message level is below minimum
        if log_levels.get(level, 1) < min_level:
            return

        # Color by level
        if level == "ERROR":
            color = "red"
        elif level == "WARNING":
            color = "orange"
        elif level == "SUCCESS":
            color = "green"
        elif level == "DEBUG":
            color = "gray"
        else:
            color = "black"

        html = f'<span style="color:{color}">[{level}] {message}</span>'
        self.log_text.append(html)

        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _on_training_completed(self, results: dict):
        """Handle training completion."""
        self.results = results

        self._log("SUCCESS", "Training completed successfully!")

        # Update results table
        self.results_table.setRowCount(0)

        for algo in ['xgboost', 'neural_network', 'ensemble']:
            if algo in results and isinstance(results[algo], dict):
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)

                algo_results = results[algo]

                self.results_table.setItem(row, 0, QTableWidgetItem(algo.replace('_', ' ').title()))
                self.results_table.setItem(row, 1, QTableWidgetItem(f"{algo_results.get('test_accuracy', 0):.1%}"))
                self.results_table.setItem(row, 2, QTableWidgetItem(f"{algo_results.get('test_accuracy_yellow', 0):.1%}"))
                self.results_table.setItem(row, 3, QTableWidgetItem(f"{algo_results.get('test_accuracy_green', 0):.1%}"))
                self.results_table.setItem(row, 4, QTableWidgetItem(f"{algo_results.get('test_accuracy_purple', 0):.1%}"))
                self.results_table.setItem(row, 5, QTableWidgetItem(algo_results.get('model_path', '')))

        self.results_table.resizeColumnsToContents()

        # Show summary
        best_model = results.get('best_model', 'unknown')
        best_accuracy = results.get('best_test_accuracy', 0)
        self._log("INFO", f"Best model: {best_model}")
        self._log("INFO", f"Best test accuracy: {best_accuracy:.1%}")

        # Re-enable start button
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.overall_progress.setValue(100)

        QMessageBox.information(self, "Training Complete",
                               f"Training completed successfully!\n\n"
                               f"Best model: {best_model}\n"
                               f"Test accuracy: {best_accuracy:.1%}\n\n"
                               f"Check the Results tab for per-zone details.")

    def _on_training_failed(self, error: str):
        """Handle training failure."""
        self._log("ERROR", f"Training failed: {error}")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        QMessageBox.critical(self, "Training Failed",
                            f"Training failed with error:\n\n{error}")
