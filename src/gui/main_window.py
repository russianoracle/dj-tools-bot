"""Main application window for Mood Classifier."""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QProgressBar,
    QFileDialog, QLabel, QMessageBox, QHeaderView, QMenu, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent

from ..audio import AudioLoader, FeatureExtractor
from ..classification import EnergyZoneClassifier, EnergyZone
from ..metadata import MetadataWriter, MetadataReader
from ..utils import get_logger

logger = get_logger(__name__)


class ProcessingThread(QThread):
    """Worker thread for processing audio files."""

    progress = pyqtSignal(int, int)  # current, total
    file_processed = pyqtSignal(str, object, object)  # filename, result, features
    finished = pyqtSignal()
    error = pyqtSignal(str, str)  # filename, error message

    def __init__(self, files, config):
        super().__init__()
        self.files = files
        self.config = config
        self.should_stop = False

    def run(self):
        """Process files in background."""
        audio_loader = AudioLoader(self.config.get('audio.sample_rate', 22050))
        feature_extractor = FeatureExtractor(self.config)
        classifier = EnergyZoneClassifier(self.config)

        total = len(self.files)

        for i, file_path in enumerate(self.files):
            if self.should_stop:
                break

            try:
                # Load audio
                y, sr = audio_loader.load(file_path)

                # Extract features
                features = feature_extractor.extract(y, sr)

                # Classify
                result = classifier.classify(features)

                # Emit result
                self.file_processed.emit(
                    Path(file_path).name,
                    result,
                    features
                )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.error.emit(Path(file_path).name, str(e))

            self.progress.emit(i + 1, total)

        self.finished.emit()

    def stop(self):
        """Stop processing."""
        self.should_stop = True


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processing_thread = None
        self.results = []

        self.setup_ui()
        self.setup_components()

        # Enable drag and drop
        self.setAcceptDrops(True)

    def setup_ui(self):
        """Set up user interface."""
        self.setWindowTitle("Mood Classifier - DJ Track Energy Zone Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("🎵 Mood Classifier - Energy Zone Analyzer")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Buttons
        button_layout = QHBoxLayout()

        self.add_files_btn = QPushButton("Add Files")
        self.add_files_btn.clicked.connect(self.add_files)
        button_layout.addWidget(self.add_files_btn)

        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.add_folder)
        button_layout.addWidget(self.add_folder_btn)

        self.process_btn = QPushButton("Analyze Tracks")
        self.process_btn.clicked.connect(self.process_files)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_table)
        button_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        self.write_metadata_btn = QPushButton("Write Metadata")
        self.write_metadata_btn.clicked.connect(self.write_metadata)
        self.write_metadata_btn.setEnabled(False)
        button_layout.addWidget(self.write_metadata_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Info label
        self.info_label = QLabel("Drag and drop audio files or folders here")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("padding: 10px; color: gray;")
        layout.addWidget(self.info_label)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            'File', 'Zone', 'Confidence', 'Tempo (BPM)',
            'Energy Var', 'Drop Int', 'Method'
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.table)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status bar
        self.statusBar().showMessage("Ready")

    def setup_components(self):
        """Set up application components."""
        self.metadata_writer = MetadataWriter(self.config)
        self.metadata_reader = MetadataReader(self.config)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        files = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file() and AudioLoader.is_supported_format(str(path)):
                files.append(str(path))
            elif path.is_dir():
                # Add all audio files from directory
                for ext in AudioLoader.SUPPORTED_FORMATS:
                    files.extend(str(f) for f in path.rglob(f'*{ext}'))

        if files:
            self.add_files_to_table(files)
            self.info_label.setText(f"{len(files)} file(s) added")
        else:
            self.info_label.setText("No supported audio files found")

    def add_files(self):
        """Add individual files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.mp4);;All Files (*)"
        )
        if files:
            self.add_files_to_table(files)

    def add_folder(self):
        """Add all files from a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            files = []
            for ext in AudioLoader.SUPPORTED_FORMATS:
                files.extend(str(f) for f in Path(folder).rglob(f'*{ext}'))

            if files:
                self.add_files_to_table(files)
                self.info_label.setText(f"Added {len(files)} files from folder")
            else:
                self.info_label.setText("No audio files found in folder")

    def add_files_to_table(self, files):
        """Add files to processing table."""
        for file_path in files:
            # Check if already in table
            existing = False
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).data(Qt.UserRole) == file_path:
                    existing = True
                    break

            if not existing:
                row = self.table.rowCount()
                self.table.insertRow(row)

                # File name
                item = QTableWidgetItem(Path(file_path).name)
                item.setData(Qt.UserRole, file_path)
                self.table.setItem(row, 0, item)

                # Status
                self.table.setItem(row, 1, QTableWidgetItem("Pending"))

        self.process_btn.setEnabled(self.table.rowCount() > 0)
        self.statusBar().showMessage(f"{self.table.rowCount()} file(s) ready for processing")

    def process_files(self):
        """Start processing files."""
        files = []
        for row in range(self.table.rowCount()):
            file_path = self.table.item(row, 0).data(Qt.UserRole)
            files.append(file_path)

        if not files:
            return

        # Disable buttons during processing
        self.process_btn.setEnabled(False)
        self.add_files_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(files))

        # Start processing thread
        self.processing_thread = ProcessingThread(files, self.config)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.file_processed.connect(self.handle_result)
        self.processing_thread.error.connect(self.handle_error)
        self.processing_thread.finished.connect(self.processing_complete)
        self.processing_thread.start()

        self.statusBar().showMessage("Processing...")

    def update_progress(self, current, total):
        """Update progress bar."""
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"Processing {current}/{total}...")

    def handle_result(self, filename, result, features):
        """Handle processing result."""
        # Find row for this file
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == filename:
                # Update row with results
                zone_item = QTableWidgetItem(f"{result.zone.emoji} {result.zone.display_name}")

                # Color code by zone
                if result.zone == EnergyZone.YELLOW:
                    zone_item.setBackground(QColor(255, 255, 200))
                elif result.zone == EnergyZone.GREEN:
                    zone_item.setBackground(QColor(200, 255, 200))
                elif result.zone == EnergyZone.PURPLE:
                    zone_item.setBackground(QColor(230, 200, 255))

                self.table.setItem(row, 1, zone_item)
                self.table.setItem(row, 2, QTableWidgetItem(f"{result.confidence:.1%}"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{features.tempo:.1f}"))
                self.table.setItem(row, 4, QTableWidgetItem(f"{features.energy_variance:.3f}"))
                self.table.setItem(row, 5, QTableWidgetItem(f"{features.drop_intensity:.2f}"))
                self.table.setItem(row, 6, QTableWidgetItem(result.method))

                # Store result
                file_path = self.table.item(row, 0).data(Qt.UserRole)
                self.results.append({
                    'file_path': file_path,
                    'filename': filename,
                    'result': result,
                    'features': features
                })

                break

    def handle_error(self, filename, error):
        """Handle processing error."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == filename:
                error_item = QTableWidgetItem(f"Error: {error}")
                error_item.setBackground(QColor(255, 200, 200))
                self.table.setItem(row, 1, error_item)
                break

    def processing_complete(self):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.export_btn.setEnabled(len(self.results) > 0)
        self.write_metadata_btn.setEnabled(len(self.results) > 0)

        # Show summary
        from collections import Counter
        zone_counts = Counter(r['result'].zone for r in self.results)

        summary = "Processing complete!\n\n"
        for zone, count in zone_counts.items():
            pct = count / len(self.results) * 100 if self.results else 0
            summary += f"{zone.emoji} {zone.display_name}: {count} ({pct:.1f}%)\n"

        self.statusBar().showMessage(f"Processed {len(self.results)} files")
        QMessageBox.information(self, "Processing Complete", summary)

    def clear_table(self):
        """Clear results table."""
        self.table.setRowCount(0)
        self.results = []
        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.write_metadata_btn.setEnabled(False)
        self.info_label.setText("Drag and drop audio files or folders here")
        self.statusBar().showMessage("Ready")

    def export_csv(self):
        """Export results to CSV."""
        if not self.results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "classification_results.csv",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'File', 'Path', 'Zone', 'Confidence', 'Tempo (BPM)',
                        'Energy Variance', 'Drop Intensity', 'Method'
                    ])

                    for r in self.results:
                        writer.writerow([
                            r['filename'],
                            r['file_path'],
                            r['result'].zone.display_name,
                            f"{r['result'].confidence:.2%}",
                            f"{r['features'].tempo:.1f}",
                            f"{r['features'].energy_variance:.4f}",
                            f"{r['features'].drop_intensity:.2f}",
                            r['result'].method
                        ])

                self.statusBar().showMessage(f"Results exported to {file_path}")
                QMessageBox.information(self, "Export Complete", f"Results saved to:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{e}")

    def write_metadata(self):
        """Write classification results to file metadata."""
        if not self.results:
            return

        reply = QMessageBox.question(
            self,
            "Write Metadata",
            f"Write classification results to {len(self.results)} file(s)?\n\n"
            "This will modify the metadata of your audio files.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            success_count = 0
            for r in self.results:
                if self.metadata_writer.write(r['file_path'], r['result']):
                    success_count += 1

            self.statusBar().showMessage(f"Metadata written to {success_count}/{len(self.results)} files")
            QMessageBox.information(
                self,
                "Metadata Written",
                f"Successfully wrote metadata to {success_count} out of {len(self.results)} files"
            )

    def show_context_menu(self, position):
        """Show context menu for table row."""
        menu = QMenu()

        view_details = QAction("View Details", self)
        view_details.triggered.connect(self.view_row_details)
        menu.addAction(view_details)

        menu.exec_(self.table.viewport().mapToGlobal(position))

    def view_row_details(self):
        """View detailed information for selected row."""
        row = self.table.currentRow()
        if row < 0:
            return

        # Find result for this row
        filename = self.table.item(row, 0).text()
        result_data = next((r for r in self.results if r['filename'] == filename), None)

        if result_data:
            features = result_data['features']
            result = result_data['result']

            details = f"File: {filename}\n\n"
            details += f"Classification: {result.zone.emoji} {result.zone.display_name}\n"
            details += f"Confidence: {result.confidence:.2%}\n"
            details += f"Method: {result.method}\n\n"
            details += "Features:\n"
            details += f"  Tempo: {features.tempo:.1f} BPM (confidence: {features.tempo_confidence:.2f})\n"
            details += f"  Zero Crossing Rate: {features.zero_crossing_rate:.4f}\n"
            details += f"  Low Energy: {features.low_energy:.2%}\n"
            details += f"  RMS Energy: {features.rms_energy:.4f}\n"
            details += f"  Spectral Rolloff: {features.spectral_rolloff:.1f} Hz\n"
            details += f"  Brightness: {features.brightness:.2%}\n"
            details += f"  Spectral Centroid: {features.spectral_centroid:.1f} Hz\n"
            details += f"  Energy Variance: {features.energy_variance:.4f}\n"
            details += f"  Drop Intensity: {features.drop_intensity:.2f}\n"

            QMessageBox.information(self, "Track Details", details)
