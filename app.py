import gradio as gr
import time
import psutil
import pandas as pd
import polars as pl
import duckdb
import dask.dataframe as dd
import gdown
import os
import matplotlib.pyplot as plt

from clearml import Task

# Set up ClearML configuration with access key and secret key
Task.set_credentials(
    api_host="https://api.clear.ml",  # Replace with your ClearML server URL
    web_host="https://app.clear.ml",  # Replace with your ClearML Web URL
    files_host="https://files.clear.ml",  # Replace with your ClearML Files URL
    key=os.environ.get('CLEARML_API_KEY'),  # Your ClearML Access Key
    secret=os.environ.get('CLEARML_API_SECRET')  # Your ClearML Secret Key
)

# Create a ClearML task
task = Task.init(project_name="nyc_taxi_trip", task_name="MyTask")

def preprocess_file(file):
    file_path = file.name

    execution_times = {}
    cpu_usages = {}
    memory_usages = {}
    libraries = {
        "pandas": pd.read_parquet,
        "polars": pl.read_parquet,
        "duckdb": lambda f: duckdb.read_parquet(f),
        "dask": lambda f: dd.read_parquet(f).compute()
    }

    for lib, reader in libraries.items():
        try:
            df, exec_time, cpu_usage, mem_usage = measure_performance(reader, file_path)
            execution_times[lib] = exec_time
            cpu_usages[lib] = cpu_usage
            memory_usages[lib] = mem_usage
            # Log metrics to ClearML
            task.logger.report_scalar(title=f"Execution Time ({lib})", series=f"Execution Time ({lib})", value=exec_time, iteration=0)
            task.logger.report_scalar(title=f"CPU Usage ({lib})", series=f"CPU Usage ({lib})", value=cpu_usage, iteration=0)
            task.logger.report_scalar(title=f"Memory Usage ({lib})", series=f"Memory Usage ({lib})", value=mem_usage, iteration=0)
        except Exception as e:
            execution_times[lib] = 0
            cpu_usages[lib] = 0
            memory_usages[lib] = 0
            print(f"Error processing {lib}: {e}")

    # Generate and save charts
    libs = list(libraries.keys())
    metrics = ["execution_times", "cpu_usages", "memory_usages"]
    file_paths = []

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        if metric == "execution_times":
            plt.bar(libs, [execution_times.get(lib, 0) for lib in libs], color='skyblue')
            plt.title("Execution Time (s)")
            plt.xlabel("Library")
            plt.ylabel("Time (s)")
            file_path_plot = "execution_time.png"
        elif metric == "cpu_usages":
            plt.bar(libs, [cpu_usages.get(lib, 0) for lib in libs], color='lightgreen')
            plt.title("CPU Usage (%)")
            plt.xlabel("Library")
            plt.ylabel("CPU (%)")
            file_path_plot = "cpu_usage.png"
        elif metric == "memory_usages":
            plt.bar(libs, [memory_usages.get(lib, 0) for lib in libs], color='lightcoral')
            plt.title("Memory Usage (MB)")
            plt.xlabel("Library")
            plt.ylabel("Memory (MB)")
            file_path_plot = "memory_usage.png"
        plt.savefig(file_path_plot)
        plt.close()
        file_paths.append(file_path_plot)
        # Upload plots to ClearML
        task.upload_artifact(name=file_path_plot, artifact_object=file_path_plot)

    # Additional chart to show time taken by each library to read the file
    plt.figure(figsize=(6, 4))
    plt.bar(libs, [execution_times.get(lib, 0) for lib in libs], color='lightblue')
    plt.title("Time Taken to Read File")
    plt.xlabel("Library")
    plt.ylabel("Time (s)")
    file_path_plot = "time_taken_to_read_file.png"
    plt.savefig(file_path_plot)
    plt.close()
    file_paths.append(file_path_plot)
    # Upload plots to ClearML
    task.upload_artifact(name=file_path_plot, artifact_object=file_path_plot)

    return file_paths[0], file_paths[1], file_paths[2], file_paths[3]

def measure_performance(func, *args, **kwargs):
    start_time = time.time()
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    result = func(*args, **kwargs)
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 2)
    return result, round(time.time() - start_time, 4), round(cpu_after - cpu_before, 2), round(mem_after - mem_before, 2)
#File Upload to proceed: 
iface = gr.Interface(
    fn=preprocess_file,
    inputs=[gr.File(label="Upload Parquet File")],
    outputs=[gr.Image(type="filepath"), gr.Image(type="filepath"), gr.Image(type="filepath"), gr.Image(type="filepath")],
    title="Data Processing Benchmark",
    description="Benchmark different libraries with a Parquet file (upload only)."
)



iface.launch(share=True)