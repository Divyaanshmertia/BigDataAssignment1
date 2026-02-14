from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, year, to_date, length, avg, count, coalesce, upper, trim, desc

# 1. Initialize Spark with Legacy Time Policy for inconsistent Gutenberg dates
spark = SparkSession.builder \
    .appName("GutenbergMetadataAnalysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load the dataset from HDFS
# Based on your setup: /user/hadoop/books/*.txt
path = "/user/hadoop/books/*.txt"
raw_rdd = spark.sparkContext.wholeTextFiles(path)

# Convert to DataFrame: (filename, full_text_content)
books_df = raw_rdd.toDF(["file_name", "text"])

# Clean file_name to remove the HDFS prefix
books_df = books_df.withColumn("file_name", regexp_extract(col("file_name"), "([^/]+)$", 1))

# --- Task 1: Robust Metadata Extraction ---

# Regex Patterns Explained:
# (?i) : Case-insensitive matching
# \s* : Matches any whitespace after the label
# (.+) : Captures everything until the newline
title_regex = r"(?i)Title:\s*(.+)"
date_regex  = r"(?i)Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4}|[A-Za-z]+, \d{4})"
lang_regex  = r"(?i)Language:\s*([A-Za-z\-]+)"
enc_regex   = r"(?i)Character set encoding:\s*([^\s\r\n]+)"

books_with_meta_df = books_df \
    .withColumn("title", trim(regexp_extract(col("text"), title_regex, 1))) \
    .withColumn("release_date_str", regexp_extract(col("text"), date_regex, 1)) \
    .withColumn("language", upper(trim(regexp_extract(col("text"), lang_regex, 1)))) \
    .withColumn("encoding", trim(regexp_extract(col("text"), enc_regex, 1)))

# --- Task 2: Deep Analysis ---

# Handle Date Inconsistencies: Attempting 4 common Gutenberg formats
analysis_df = books_with_meta_df.withColumn("release_year", coalesce(
    year(to_date(col("release_date_str"), "MMMM d, yyyy")),
    year(to_date(col("release_date_str"), "MMM d, yyyy")),
    year(to_date(col("release_date_str"), "MMMM, yyyy")),
    year(to_date(col("release_date_str"), "MMM, yyyy"))
))

# 1. Number of books released each year
books_per_year = analysis_df.filter(col("release_year").isNotNull()) \
    .groupBy("release_year").count().orderBy("release_year")

# 2. Most common language
common_lang = analysis_df.filter(col("language") != "") \
    .groupBy("language").count().orderBy(col("count").desc())

# 3. Average length of book titles
avg_title_row = analysis_df.filter(col("title") != "") \
    .select(avg(length(col("title"))).alias("avg_title_length")).first()
avg_title_val = avg_title_row["avg_title_length"]

# --- Calculate Specific Report Metrics ---

# A. Extraction Success Rate
total_records = books_df.count()
successful_dates = analysis_df.filter(col("release_year").isNotNull()).count()
success_rate = (successful_dates / total_records) * 100

# B. Peak Year Details
# Sort by count descending to find the top year
peak_year_row = books_per_year.orderBy(desc("count")).first()
peak_year = peak_year_row["release_year"]
peak_count = peak_year_row["count"]

# C. Top Language Details
top_lang_row = common_lang.first()
top_language = top_lang_row["language"]
top_lang_count = top_lang_row["count"]

# --- Print Results ---
print("\n" + "="*60)
print("PROJECT GUTENBERG METADATA ANALYSIS")
print("="*60)

print("\n[A] METRICS SUMMARY (For Report):")
print(f"   - Total Books Processed:    {total_records}")
print(f"   - Date Extraction Success:  {success_rate:.2f}% ({successful_dates}/{total_records})")
print(f"   - Peak Release Year:        {peak_year} ({peak_count} books)")
print(f"   - Dominant Language:        {top_language} ({top_lang_count} books)")
print(f"   - Average Title Length:     {avg_title_val:.2f} characters")

print("\n[B] DETAILED: BOOKS RELEASED PER YEAR:")
books_per_year.orderBy("release_year").show(20)

print("\n[C] DETAILED: LANGUAGE DISTRIBUTION:")
common_lang.show()

spark.stop()
