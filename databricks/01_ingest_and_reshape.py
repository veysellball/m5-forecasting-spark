# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Ingest and Reshape
# MAGIC Pipeline to ingest raw M5 CSV files and reshape them to long format.

# COMMAND ----------

import sys
REPO = "/Workspace/Users/veys61206@gmail.com/m5-forecasting-spark"
if REPO not in sys.path:
    sys.path.append(REPO)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Read the raw files

# COMMAND ----------

from pyspark.sql import functions as F

RAW = "/Volumes/workspace/m5/raw"

sales_wide = spark.read.csv(f"{RAW}/sales_train_evaluation.csv", header=True)
calendar   = spark.read.csv(f"{RAW}/calendar.csv", header=True)
prices     = spark.read.csv(f"{RAW}/sell_prices.csv", header=True)

print("Sales wide sütun sayısı:", len(sales_wide.columns))
