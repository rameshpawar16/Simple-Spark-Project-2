from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler

# -----------------------------------------
# 1. Create Spark Session
# -----------------------------------------
spark = SparkSession.builder \
    .appName("Big Data ML Preprocessing") \
    .getOrCreate()

# -----------------------------------------
# 2. Load CSV Dataset
# -----------------------------------------
df = spark.read.csv(
    "student.csv",
    header=True,
    inferSchema=True
)

print("Original Dataset:")
df.show()

# -----------------------------------------
# 3. Handle Missing Values
# -----------------------------------------
df_clean = df.fillna({
    "marks": 0,
    "attendance": 0
})

print("After Handling Missing Values:")
df_clean.show()

# -----------------------------------------
# 4. Feature Selection
# -----------------------------------------
selected_df = df_clean.select("age", "marks", "attendance")

print("Selected Features:")
selected_df.show()

# -----------------------------------------
# 5. Convert to ML Ready Format
# -----------------------------------------
assembler = VectorAssembler(
    inputCols=["age", "marks", "attendance"],
    outputCol="features"
)

ml_ready_df = assembler.transform(selected_df)

print("ML Ready Dataset:")
ml_ready_df.select("features").show()

# -----------------------------------------
# 6. Save Processed Dataset
# -----------------------------------------
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# Convert vector to array
array_df = ml_ready_df.withColumn("features_array", vector_to_array(col("features")))

# Extract individual columns
final_df = array_df.select(
    col("features_array")[0].alias("age"),
    col("features_array")[1].alias("marks"),
    col("features_array")[2].alias("attendance")
)

# Save dataset
final_df.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("ml_ready_output")

print("Processed dataset saved successfully.")

spark.stop()