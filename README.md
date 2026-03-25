# radar_camera_sync

## Project Overview
The Radar–Camera Synchronisation Dashboard is a tool designed to visualize and analyze the synchronization between radar and camera sensors. It provides playback controls, radar and camera visualizations, and quality metrics to evaluate the alignment of sensor data. The application has been migrated from a Streamlit-based web dashboard to a PyQt-based GUI for enhanced user experience.

## Project Structure
```
├── app_pyqt.py                # Main PyQt application
├── app_streamlit.py           # Original Streamlit application (deprecated)
├── config.yaml                # Configuration file for the application
├── requirements.txt           # Python dependencies
├── backend/                   # Backend logic for data processing
│   ├── alignment_engine.py    # Handles radar-camera alignment
│   ├── camera_renderer.py     # Renders camera frames
│   ├── data_loader.py         # Loads sensor data
│   ├── metadata_parser.py     # Parses metadata
│   ├── metrics_engine.py      # Computes synchronization metrics
│   ├── playback_engine.py     # Manages playback functionality
│   ├── radar_renderer.py      # Renders radar visualizations
│   ├── validation.py          # Validates data integrity
├── frontend/                  # Frontend logic for the Streamlit app
│   ├── app.py                 # Streamlit app logic
│   ├── callbacks.py           # Streamlit callbacks
│   ├── layouts.py             # Streamlit layouts
├── tests/                     # Unit tests for the application
│   ├── test_alignment.py      # Tests for alignment engine
│   ├── test_components.py     # Tests for components
│   ├── test_integration.py    # Integration tests
```

## Features
- **Playback Controls**: Scrub through data, play/pause, and adjust playback speed.
- **Radar View Selector**: Choose from multiple radar visualizations (e.g., bird's-eye view, range-doppler).
- **Quality Metrics**: Evaluate synchronization quality with metrics like mean offset, jitter, and drift.
- **Side-by-Side Visualizations**: Compare radar and camera data in real-time.
- **Live Sync Indicator**: Visualize frame offsets dynamically.

## How to Run
### Prerequisites
- Python 3.8 or higher
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application
1. Navigate to the project directory:

```bash
cd /home/user/Music/sync_dashboard
```

2. Run the PyQt application:

```bash
python app_pyqt.py
```

3. The GUI will launch, allowing you to load data and interact with the dashboard.

### Optional: Running the Streamlit App (Deprecated)
If you want to run the original Streamlit application:

```bash
streamlit run app_streamlit.py
```

## Configuration
The application uses a `config.yaml` file for settings such as frame drop thresholds, quality evaluation parameters, and radar configurations. Modify this file to customize the application's behavior.


## License
This project is licensed under the MIT License.
