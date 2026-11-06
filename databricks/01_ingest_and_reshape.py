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
