# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "statsforecast",
# ]
# ///
# MAGIC %md
# MAGIC # Step 3: Ingest and Reshape
# MAGIC Pipeline to ingest raw M5 CSV files, unpivot to long format, and build the base table.

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

print("Sales wide column count:", len(sales_wide.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Wide → long with unpivot

# COMMAND ----------

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
D_COLS = [c for c in sales_wide.columns if c.startswith("d_")]

sales_long = (
    sales_wide
    .unpivot(ID_COLS, D_COLS, "d", "sales")
    .withColumn("d_num", F.expr("CAST(substr(d, 3) AS INT)"))
    .withColumn("sales", F.col("sales").cast("int"))
    .drop("d")
)

calendar = (
    calendar
    .withColumn("d_num", F.expr("CAST(substr(d, 3) AS INT)"))
    .withColumn("date", F.to_date("date"))
    .withColumn("wm_yr_wk", F.col("wm_yr_wk").cast("int"))
    .withColumn("snap_CA", F.col("snap_CA").cast("int"))
    .withColumn("snap_TX", F.col("snap_TX").cast("int"))
    .withColumn("snap_WI", F.col("snap_WI").cast("int"))
    .drop("d")
)

prices = (
    prices
    .withColumn("wm_yr_wk", F.col("wm_yr_wk").cast("int"))
    .withColumn("sell_price", F.col("sell_price").cast("double"))
)

sales_long.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Lazy evaluation

# COMMAND ----------

import time

t = time.time()
x = sales_long.filter("store_id = 'CA_1'").withColumn("double_sales", F.col("sales") * 2)
print(f"Defining the transformation: {time.time() - t:.3f}s")

t = time.time()
print("Row count for CA_1:", x.count())
print(f"The action (count): {time.time() - t:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Write the bronze tables and validate

# COMMAND ----------

sales_long.write.mode("overwrite").saveAsTable("workspace.m5.sales_long")
calendar.write.mode("overwrite").saveAsTable("workspace.m5.calendar")
prices.write.mode("overwrite").saveAsTable("workspace.m5.prices")

sl = spark.table("workspace.m5.sales_long")
n = sl.count()
assert n == 30_490 * 1_941, f"Expected 59,181,090 rows, got {n}"
assert sl.filter("sales IS NULL OR sales < 0").count() == 0, "Found invalid sales"
assert sl.select("id").distinct().count() == 30_490, "Distinct id count mismatch"
print(f"{n:,} rows validated successfully ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Build the base table

# COMMAND ----------

import sys
if REPO not in sys.path:
    sys.path.append(REPO)

from src.features import build_base

base = build_base(
    spark.table("workspace.m5.sales_long"),
    spark.table("workspace.m5.calendar"),
    spark.table("workspace.m5.prices")
)
base.write.mode("overwrite").saveAsTable("workspace.m5.base")

b = spark.table("workspace.m5.base")
print(f"{b.count():,} rows after dropping pre-launch days")

# Contiguity: from launch on, every series must have every day
gaps = (
    b.groupBy("id")
    .agg(
        F.count("*").alias("n"),
        F.min("d_num").alias("lo"),
        F.max("d_num").alias("hi")
    )
    .filter("n != hi - lo + 1")
)
assert gaps.count() == 0, "Some series have missing days after launch — lags would be wrong"
print("Contiguity check passed ✓")