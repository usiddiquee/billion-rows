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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, MaxAbsScaler, PolynomialFeatures
from scipy.stats import zscore, boxcox
from sklearn.decomposition import PCA

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

def perform_action(file, column, action, normalization, encoding, outlier_method, transformation, feature_engineering, data_reduction):
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
    
    con = None  # Reuse the connection for DuckDB
    
    for lib, reader in libraries.items():
        try:
            start_time = time.time()
            process = psutil.Process()
            cpu_before = process.cpu_percent(interval=None)
            mem_before = process.memory_info().rss / (1024 ** 2)  # Memory in MB
            
            file_reading_start_time = time.time()
            if lib == "duckdb":
                if con is None:
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
                    elif normalization == "Robust Scaler":
                        scaler = RobustScaler()
                        df[column] = scaler.fit_transform(df[[column]])
                    elif normalization == "Max Abs Scaler":
                        scaler = MaxAbsScaler()
                        df[column] = scaler.fit_transform(df[[column]])
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
            
            elif action == "Encode Categorical Variables":
                if encoding == "One-Hot Encoding":
                    if lib == "pandas":
                        df = pd.get_dummies(df, columns=[column])
                    elif lib == "polars":
                        df = df.to_dummies(columns=[column])
                    elif lib == "duckdb":
                        df = pd.get_dummies(df, columns=[column])
                    elif lib == "dask":
                        df = dd.get_dummies(df, columns=[column])
                elif encoding == "Label Encoding":
                    if lib == "pandas":
                        df[column] = LabelEncoder().fit_transform(df[column])
                    elif lib == "polars":
                        df = df.with_columns([pl.col(column).cast(pl.Categorical).cast(pl.Int32)])
                    elif lib == "duckdb":
                        df[column] = LabelEncoder().fit_transform(df[column])
                    elif lib == "dask":
                        df[column] = dd.from_pandas(LabelEncoder().fit_transform(df[column].compute()), npartitions=1)
            
            elif action == "Handle Outliers":
                if outlier_method == "Z-Score Method":
                    if lib == "pandas":
                        df = df[(np.abs(zscore(df[column])) < 3)]
                    elif lib == "polars":
                        df = df.filter(np.abs(zscore(df[column])) < 3)
                    elif lib == "duckdb":
                        df = df[(np.abs(zscore(df[column])) < 3)]
                    elif lib == "dask":
                        df = df[(np.abs(zscore(df[column].compute())) < 3)]
                elif outlier_method == "IQR Method":
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    if lib == "pandas":
                        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
                    elif lib == "polars":
                        df = df.filter(~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))))
                    elif lib == "duckdb":
                        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
                    elif lib == "dask":
                        df = df[~((df[column].compute() < (Q1 - 1.5 * IQR)) | (df[column].compute() > (Q3 + 1.5 * IQR)))]
            
            elif action == "Data Transformation":
                if transformation == "Log Transformation":
                    if lib == "pandas":
                        df[column] = np.log1p(df[column])
                    elif lib == "polars":
                        df = df.with_columns([pl.col(column).log1p().alias(column)])
                    elif lib == "duckdb":
                        df[column] = np.log1p(df[column])
                    elif lib == "dask":
                        df[column] = dd.from_pandas(np.log1p(df[column].compute()), npartitions=1)
                elif transformation == "Box-Cox Transformation":
                    if lib == "pandas":
                        df[column], _ = boxcox(df[column])
                    elif lib == "polars":
                        df = df.with_columns([pl.col(column).apply(lambda x: boxcox(x)[0]).alias(column)])
                    elif lib == "duckdb":
                        df[column], _ = boxcox(df[column])
                    elif lib == "dask":
                        df[column] = dd.from_pandas(boxcox(df[column].compute())[0], npartitions=1)
            
            elif action == "Feature Engineering":
                if feature_engineering == "Polynomial Features":
                    if lib == "pandas":
                        poly = PolynomialFeatures(degree=2)
                        df = pd.DataFrame(poly.fit_transform(df[[column]]), columns=poly.get_feature_names_out([column]))
                    elif lib == "polars":
                        df = df.with_columns([pl.col(column).pow(2).alias(f"{column}_squared")])
                    elif lib == "duckdb":
                        df[column + "_squared"] = df[column] ** 2
                    elif lib == "dask":
                        df[column + "_squared"] = df[column] ** 2
                elif feature_engineering == "Interaction Terms":
                    if lib == "pandas":
                        df[column + "_interaction"] = df[column] * df[column]
                    elif lib == "polars":
                        df = df.with_columns([pl.col(column).mul(pl.col(column)).alias(f"{column}_interaction")])
                    elif lib == "duckdb":
                        df[column + "_interaction"] = df[column] * df[column]
                    elif lib == "dask":
                        df[column + "_interaction"] = df[column] * df[column]
            
            elif action == "Data Reduction":
                if data_reduction == "PCA":
                    if lib == "pandas":
                        pca = PCA(n_components=1)
                        df[column + "_pca"] = pca.fit_transform(df[[column]])
                    elif lib == "polars":
                        pca = PCA(n_components=1)
                        df = df.with_columns([pl.Series(pca.fit_transform(df[[column]]).alias(f"{column}_pca"))])
                    elif lib == "duckdb":
                        pca = PCA(n_components=1)
                        df[column + "_pca"] = pca.fit_transform(df[[column]])
                    elif lib == "dask":
                        pca = PCA(n_components=1)
                        df[column + "_pca"] = dd.from_pandas(pca.fit_transform(df[[column]].compute()), npartitions=1)
                elif data_reduction == "Feature Selection":
                    if lib == "pandas":
                        corr = df.corr()
                        df = df[corr[column].abs().sort_values(ascending=False).index[:5]]
                    elif lib == "polars":
                        corr = df.corr()
                        df = df.select(corr[column].abs().sort(reverse=True).columns[:5])
                    elif lib == "duckdb":
                        corr = df.corr()
                        df = df[corr[column].abs().sort_values(ascending=False).index[:5]]
                    elif lib == "dask":
                        corr = df.corr().compute()
                        df = df[corr[column].abs().sort_values(ascending=False).index[:5]]
            
            cpu_after = process.cpu_percent(interval=None)
            mem_after = process.memory_info().rss / (1024 ** 2)  # Memory in MB
            
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
    info_button = gr.Button("Get Data Info")
    info_text = gr.Textbox(label="Data Information")
    column_input = gr.Textbox(label="Enter Column Name")
    action_input = gr.Dropdown(label="Select Action", choices=["Handle Missing Values", "Normalize Data", "Encode Categorical Variables", "Handle Outliers", "Data Transformation", "Feature Engineering", "Data Reduction"])
    normalization_input = gr.Dropdown(label="Select Normalization Method", choices=["Min-Max Scaler", "Standard Scaler", "Robust Scaler", "Max Abs Scaler"], visible=False)
    encoding_input = gr.Dropdown(label="Select Encoding Method", choices=["One-Hot Encoding", "Label Encoding"], visible=False)
    outlier_input = gr.Dropdown(label="Select Outlier Handling Method", choices=["Z-Score Method", "IQR Method"], visible=False)
    transformation_input = gr.Dropdown(label="Select Transformation Method", choices=["Log Transformation", "Box-Cox Transformation"], visible=False)
    feature_engineering_input = gr.Dropdown(label="Select Feature Engineering Method", choices=["Polynomial Features", "Interaction Terms"], visible=False)
    data_reduction_input = gr.Dropdown(label="Select Data Reduction Method", choices=["PCA", "Feature Selection"], visible=False)
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
        lambda x: gr.update(visible=x == "Normalize Data"), inputs=[action_input], outputs=[normalization_input]
    )

    action_input.change(
        lambda x: gr.update(visible=x == "Encode Categorical Variables"), inputs=[action_input], outputs=[encoding_input]
    )

    action_input.change(
        lambda x: gr.update(visible=x == "Handle Outliers"), inputs=[action_input], outputs=[outlier_input]
    )

    action_input.change(
        lambda x: gr.update(visible=x == "Data Transformation"), inputs=[action_input], outputs=[transformation_input]
    )

    action_input.change(
        lambda x: gr.update(visible=x == "Feature Engineering"), inputs=[action_input], outputs=[feature_engineering_input]
    )

    action_input.change(
        lambda x: gr.update(visible=x == "Data Reduction"), inputs=[action_input], outputs=[data_reduction_input]
    )

    action_button.click(
        perform_action,
        inputs=[file_upload, column_input, action_input, normalization_input, encoding_input, outlier_input, transformation_input, feature_engineering_input, data_reduction_input],
        outputs=[results_text, execution_time_image, cpu_usage_image, memory_usage_image, file_reading_time_image]
    )

demo.launch(share=True)