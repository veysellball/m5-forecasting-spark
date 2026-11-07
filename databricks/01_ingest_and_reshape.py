# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2.5: Smoke Test & Step 3: Ingest and Reshape

# COMMAND ----------

import sys
REPO = "/Workspace/Users/veys61206@gmail.com/m5-forecasting-spark"
if REPO not in sys.path:
    sys.path.append(REPO)

# COMMAND ----------

# 1. Spark versiyonu
print("Databricks Spark Version:", spark.version)

# 2. Volume'deki takvim verisini okuma testi
cal = spark.read.csv("/Volumes/workspace/m5/raw/calendar.csv", header=True)
print("Calendar satır sayısı:", cal.count())  # 1969 çıkmalı

# COMMAND ----------

# 3. Serverless compute üzerinde applyInPandas testi
def first_row(pdf):
    return pdf.head(1)[["weekday", "d"]]

cal.groupBy("weekday").applyInPandas(first_row, "weekday string, d string").show()

# COMMAND ----------

# MAGIC %pip install statsforecast



# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3.1: Read the raw files
# COMMAND ----------
from pyspark.sql import functions as F
RAW = "/Volumes/workspace/m5/raw"
sales_wide = spark.read.csv(f"{RAW}/sales_train_evaluation.csv", header=True)
calendar   = spark.read.csv(f"{RAW}/calendar.csv", header=True)
prices     = spark.read.csv(f"{RAW}/sell_prices.csv", header=True)
print("Sales wide sütun sayısı:", len(sales_wide.columns))