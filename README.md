---
title: Billion Row Challenge _ NYC Taxi Data Processing Benchmark
sdk: gradio
emoji: 📚
colorFrom: blue
colorTo: red
---


## Data Processing and Benchmarking Tool

This project provides an interactive web application built using Gradio for benchmarking various Python libraries (Pandas, Polars, DuckDB, and Dask) for NYC Taxi Trip data processing. It leverages ClearML for experiment tracking and logging.

## Features
- **Data Information Display:** Upload Parquet files and display dataset information such as shape, columns, data types, and statistics.
- **Benchmarking Data Processing Libraries:** Compare the performance of Pandas, Polars, DuckDB, and Dask in handling missing values and normalizing data.
- **Visualization of Results:** Generate bar charts showing execution time, CPU usage, memory usage, and file reading time for each library.
- **ClearML Integration:** Log metrics and upload generated artifacts to ClearML for tracking and analysis.

## Installation
1. Clone the repository.
2. Install dependencies using:
   ```sh
   pip install gradio psutil pandas polars duckdb dask matplotlib clearml numpy
   ```

3. Set ClearML environment variables:
   ```sh
   export CLEARML_API_KEY="your_api_key"
   export CLEARML_API_SECRET="your_api_secret"
   ```

## Usage
Run the application with:
```sh
python app.py
```
This will launch the Gradio app with a shareable link.

## Code Highlights
- **Library Benchmarking:** Measures time, CPU, and memory usage for data operations.
- **Dynamic UI:** Gradio components update based on user selections.
- **ClearML Logging:** Logs all metrics and uploads artifacts for easy monitoring.

## Directory Structure
```
project-directory/
│
├── app.py               # Main application code
├── requirements.txt     # List of dependencies
└── README.md            # This readme file
```

## Future Enhancements
- Add support for more preprocessing operations.
- Enable batch processing for large datasets.
- Implement additional machine learning workflows.

## License
This project is licensed under the MIT License.



## Author
Umar Ahmad Siddiquee

---

This project provides a streamlined way to benchmark data processing libraries with ClearML integration and Gradio-based visualization.

