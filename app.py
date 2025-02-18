import gradio as gr
import time
import psutil
import pandas as pd
import polars as pl
import duckdb
import dask.dataframe as dd
import matplotlib.pyplot as plt
from clearml import Task
import os
import numpy as np

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

def display_data_info(file):
    # Read the uploaded Parquet file using DuckDB
    con = duckdb.connect(':memory:')
    df = con.execute("SELECT * FROM read_parquet('" + file.name + "')").fetchdf()
    
    # Get information about the data
    info = df.describe()
    columns = df.columns.tolist()
    dtypes = df.dtypes.tolist()
    shape = df.shape
    
    # Create a string to display the information
    info_str = "Data Shape: {}\n".format(shape)
    info_str += "Columns: {}\n".format(columns)
    info_str += "Data Types: {}\n\n".format(dtypes)
    info_str += str(info)
    
    return info_str, columns

def perform_action(file, column, action, normalization):
    # Read the uploaded Parquet file using different libraries
    libraries = {
        "pandas": pd.read_parquet,
        "polars": pl.read_parquet,
        "duckdb": lambda f: duckdb.connect(':memory:').execute("SELECT * FROM read_parquet('" + f.name + "')").fetchdf(),
        "dask": lambda f: dd.read_parquet(f.name).compute()
    }
    
    execution_times = {}
    cpu_usages = {}
    memory_usages = {}
    file_reading_times = {}
    
    for lib, reader in libraries.items():
        try:
            start_time = time.time()
            process = psutil.Process()
            cpu_before = process.cpu_percent(interval=None)
            mem_before = process.memory_info().rss / (1024 ** 2)  # Memory in MB
            
            file_reading_start_time = time.time()
            if lib == "duckdb":
                con = duckdb.connect(':memory:')
                df = con.execute("SELECT * FROM read_parquet('" + file.name + "')").fetchdf()
            else:
                df = reader(file)
            file_reading_end_time = time.time()
            file_reading_times[lib] = round(file_reading_end_time - file_reading_start_time, 4)
            
            if action == "Handle Missing Values":
                if lib == "pandas":
                    df[column] = df[column].fillna(df[column].mean())
                elif lib == "polars":
                    df = df.fill_null(column, df[column].mean())
                elif lib == "duckdb":
                    df = df.fillna({column: df[column].mean()})
                elif lib == "dask":
                    df = df.fillna({column: df[column].mean()})
            
            elif action == "Normalize Data":
                if lib == "pandas":
                    if normalization == "Min-Max Scaler":
                        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                    elif normalization == "Standard Scaler":
                        df[column] = (df[column] - df[column].mean()) / df[column].std()
                elif lib == "polars":
                    if normalization == "Min-Max Scaler":
                        df = df.with_columns([(df[column] - df[column].min()) / (df[column].max() - df[column].min()).alias(column)])
                    elif normalization == "Standard Scaler":
                        df = df.with_columns([(df[column] - df[column].mean()) / df[column].std().alias(column)])
                elif lib == "duckdb":
                    if normalization == "Min-Max Scaler":
                        df = df.assign(**{column: (df[column] - df[column].min()) / (df[column].max() - df[column].min())})
                    elif normalization == "Standard Scaler":
                        df = df.assign(**{column: (df[column] - df[column].mean()) / df[column].std()})
                elif lib == "dask":
                    if normalization == "Min-Max Scaler":
                        df = df.assign(**{column: (df[column] - df[column].min()) / (df[column].max() - df[column].min())})
                    elif normalization == "Standard Scaler":
                        df = df.assign(**{column: (df[column] - df[column].mean()) / df[column].std()})
            
            cpu_after = process.cpu_percent(interval=None)
            mem_after = process.memory_info().rss / (1024 ** 2)
            peak_mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)
            
            execution_times[lib] = round(time.time() - start_time, 4)
            cpu_usages[lib] = round(cpu_after - cpu_before, 2)
            memory_usages[lib] = round(mem_after - mem_before, 2)
            
            # Log metrics to ClearML
            task.logger.report_scalar(title=f"Execution Time ({lib})", series=f"Execution Time ({lib})", value=execution_times[lib], iteration=0)
            task.logger.report_scalar(title=f"CPU Usage ({lib})", series=f"CPU Usage ({lib})", value=cpu_usages[lib], iteration=0)
            task.logger.report_scalar(title=f"Memory Usage ({lib})", series=f"Memory Usage ({lib})", value=memory_usages[lib], iteration=0)
        except Exception as e:
            execution_times[lib] = 0
            cpu_usages[lib] = 0
            memory_usages[lib] = 0
            file_reading_times[lib] = 0
            print(f"Error processing {lib}: {e}")
    
    # Generate and save charts
    libs = list(libraries.keys())
    plt.figure(figsize=(6, 4))
    plt.bar(libs, [execution_times.get(lib, 0) for lib in libs], color='skyblue')
    plt.title("Execution Time (s)")
    plt.xlabel("Library")
    plt.ylabel("Time (s)")
    plt.savefig("execution_time.png")
    plt.close()
    
    plt.figure(figsize=(6, 4))
    plt.bar(libs, [cpu_usages.get(lib, 0) for lib in libs], color='lightgreen')
    plt.title("CPU Usage (%)")
    plt.xlabel("Library")
    plt.ylabel("CPU (%)")
    plt.savefig("cpu_usage.png")
    plt.close()
    
    plt.figure(figsize=(6, 4))
    plt.bar(libs, [memory_usages.get(lib, 0) for lib in libs], color='lightcoral')
    plt.title("Memory Usage (MB)")
    plt.xlabel("Library")
    plt.ylabel("Memory (MB)")
    plt.savefig("memory_usage.png")
    plt.close()
    
    plt.figure(figsize=(6, 4))
    plt.bar(libs, [file_reading_times.get(lib, 0) for lib in libs], color='yellow')
    plt.title("File Reading Time (s)")
    plt.xlabel("Library")
    plt.ylabel("Time (s)")
    plt.savefig("file_reading_time.png")
    plt.close()
    
    # Upload plots to ClearML
    task.upload_artifact(name="execution_time.png", artifact_object="execution_time.png")
    task.upload_artifact(name="cpu_usage.png", artifact_object="cpu_usage.png")
    task.upload_artifact(name="memory_usage.png", artifact_object="memory_usage.png")
    task.upload_artifact(name="file_reading_time.png", artifact_object="file_reading_time.png")
    
    return {
        "pandas": {
            "Time Taken": execution_times.get("pandas", 0),
            "CPU Utilization": cpu_usages.get("pandas", 0),
            "Memory Usage": memory_usages.get("pandas", 0),
            "File Reading Time": file_reading_times.get("pandas", 0),
            "Result": "success"
        },
        "polars": {
            "Time Taken": execution_times.get("polars", 0),
            "CPU Utilization": cpu_usages.get("polars", 0),
            "Memory Usage": memory_usages.get("polars", 0),
            "File Reading Time": file_reading_times.get("polars", 0),
            "Result": "success"
        },
        "duckdb": {
            "Time Taken": execution_times.get("duckdb", 0),
            "CPU Utilization": cpu_usages.get("duckdb", 0),
            "Memory Usage": memory_usages.get("duckdb", 0),
            "File Reading Time": file_reading_times.get("duckdb", 0),
            "Result": "success"
        },
        "dask": {
            "Time Taken": execution_times.get("dask", 0),
            "CPU Utilization": cpu_usages.get("dask", 0),
            "Memory Usage": memory_usages.get("dask", 0),
            "File Reading Time": file_reading_times.get("dask", 0),
            "Result": "success"
        }
    }, "execution_time.png", "cpu_usage.png", "memory_usage.png", "file_reading_time.png"

demo = gr.Blocks()

with demo:
    gr.Markdown("# Data Information and Preprocessing Benchmarking")
    
    file_upload = gr.File(label="Upload Parquet File")
    info_button = gr.Button("Get Data Information")
    info_text = gr.Textbox(label="Data Information")
    column_input = gr.Textbox(label="Enter Column Name")
    action_input = gr.Dropdown(label="Select Action", choices=["Handle Missing Values", "Normalize Data"])
    normalization_input = gr.Dropdown(label="Select Normalization Method", choices=["Min-Max Scaler", "Standard Scaler"], visible=False)
    action_button = gr.Button("Perform Action")
    results_text = gr.Textbox(label="Results")
    execution_time_image = gr.Image(label="Execution Time")
    cpu_usage_image = gr.Image(label="CPU Usage")
    memory_usage_image = gr.Image(label="Memory Usage")
    file_reading_time_image = gr.Image(label="File Reading Time")

    info_button.click(
        display_data_info,
        inputs=[file_upload],
        outputs=[info_text, column_input]
    )

    action_input.change(
        lambda x: normalization_input.show() if x == "Normalize Data" else normalization_input.hide(),
        inputs=[action_input],
        outputs=[normalization_input]
    )

    action_button.click(
        perform_action,
        inputs=[file_upload, column_input, action_input, normalization_input],
        outputs=[results_text, execution_time_image, cpu_usage_image, memory_usage_image, file_reading_time_image]
    )

    action_input.change(
        lambda x: normalization_input.update(visible=True) if x == "Normalize Data" else normalization_input.update(visible=False),
        inputs=[action_input],
        outputs=[normalization_input]
    )

demo.launch(share=True)