---
title: Billion Row Challenge _ NYC Taxi Data Processing Benchmark
sdk: gradio
emoji: 📚
colorFrom: blue
colorTo: red
---


# Data Processing Benchmark

This project benchmarks the performance of different data processing libraries when reading a NYC taxi dataset from Google Drive. The results are logged to ClearML and visualized using matplotlib.

## Libraries Used
- Gradio
- time
- psutil
- pandas
- polars
- duckdb
- dask
- gdown
- matplotlib
- ClearML

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set ClearML Credentials:**
   Add your ClearML API key and secret to the environment variables:
   ```bash
   export CLEARML_API_KEY="your-api-key"
   export CLEARML_API_SECRET="your-secret-key"
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```

The Gradio interface will launch, and you can access it through the provided link.

## Key Functions
- **download_file:** Downloads the dataset from Google Drive.
- **preprocess_file:** Measures performance metrics (execution time, CPU usage, memory usage) for Pandas, Polars, DuckDB, and Dask.
- **measure_performance:** Captures execution time, CPU usage, and memory usage.

## Outputs
- Bar charts comparing execution time, CPU usage, and memory usage across libraries.
- All charts are uploaded to ClearML as artifacts.

## ClearML Integration
- Ensure your ClearML server URL, web URL, and files URL are correctly set.
- Logs performance metrics and uploads generated plots.

## Gradio Interface
- The interface displays the generated charts directly in the browser.
- Allows easy sharing of the interface with others.

## Dataset
- NYC Taxi Trip data in Parquet format, downloaded from Google Drive.

## License
MIT License

## Author
Umar Ahmad Siddiquee

---

This project provides a streamlined way to benchmark data processing libraries with ClearML integration and Gradio-based visualization.

