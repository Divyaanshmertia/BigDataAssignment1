import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, expr, udf, substring_index
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, Normalizer
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

# 1. SETUP
spark = SparkSession.builder \
    .appName("GutenbergTFIDF_Final") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
start_time = time.time()

# 2. LOAD DATA
INPUT_PATH = "file:///home/hadoop/Downloads/D184MB/*.txt"
TARGET_BOOK = "10.txt"  # King James Bible

print(f"[1/5] Loading Data from {INPUT_PATH}...")
df = spark.sparkContext.wholeTextFiles(INPUT_PATH, minPartitions=20).toDF(["file_path", "text"])
df = df.withColumn("file_name", expr("regexp_extract(file_path, '([^/]+)$', 1)"))

# 3. PREPROCESSING (Meeting ALL Requirements)
print("[2/5] Cleaning Text...")

# A. Remove Project Gutenberg Header/Footer (Fast Substring Method)
# Logic: Take text AFTER the last "*** START OF", then take text BEFORE the first "*** END OF"
# This avoids the regex "hanging" issue while fulfilling the requirement.
df_clean = df.withColumn("clean_text", substring_index(col("text"), "*** START OF", -1))
df_clean = df_clean.withColumn("clean_text", substring_index(col("clean_text"), "*** END OF", 1))

# B. Lowercase
df_clean = df_clean.withColumn("clean_text", lower(col("clean_text")))

# C. Remove Punctuation (Keep only a-z and whitespace)
df_clean = df_clean.withColumn("clean_text", regexp_replace(col("clean_text"), "[^a-z\\s]", ""))

# 4. TF-IDF PIPELINE
print("[3/5] Building TF-IDF Pipeline...")

# D. Tokenize into words
tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\s+")

# E. Remove Stop Words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# F. TF Calculation (CountVectorizer)
# vocabSize=10000 ensures memory safety
cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=10000, minDF=2.0)

# G. IDF Calculation
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

# H. Normalization (Crucial for Cosine Similarity)
normalizer = Normalizer(inputCol="tfidf_features", outputCol="norm_features", p=2.0)

pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, normalizer])

print("      > Fitting & Transforming...")
model = pipeline.fit(df_clean)
results_df = model.transform(df_clean)
results_df.cache()
results_df.count() # Force materialization

# 5. COSINE SIMILARITY
print(f"[4/5] Calculating Similarity for '{TARGET_BOOK}'...")

target_row = results_df.filter(col("file_name") == TARGET_BOOK).select("norm_features").first()
if not target_row:
    print(f"Error: {TARGET_BOOK} not found.")
    sys.exit(1)

# Broadcast the target vector for efficiency
target_vector = spark.sparkContext.broadcast(target_row["norm_features"])

# Cosine Similarity UDF
def cos_sim(v):
    if v is None: return 0.0
    return float(v.dot(target_vector.value))

sim_udf = udf(cos_sim, DoubleType())
sim_df = results_df.withColumn("similarity", sim_udf(col("norm_features")))

# 6. OUTPUT
print(f"[5/5] Top 5 Similar Books:")
top_books = sim_df.filter(col("file_name") != TARGET_BOOK) \
    .select("file_name", "similarity") \
    .orderBy(col("similarity").desc()) \
    .limit(5)

print("="*40)
top_books.show(truncate=False)
print("="*40)
print(f"Total Time: {time.time() - start_time:.2f}s")
spark.stop()
