import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, regexp_replace, year, to_date, countDistinct, desc, trim, coalesce
from pyspark.sql.types import IntegerType

# 1. INITIALIZE SPARK
spark = SparkSession.builder \
    .appName("AuthorInfluenceNetwork") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
start_time = time.time()

# 2. LOAD DATA
INPUT_PATH = "file:///home/hadoop/Downloads/D184MB/*.txt" 
X_YEARS = 5  # The Influence Window

print(f"[1/5] Loading Data from {INPUT_PATH}...")
# Min partitions ensures parallelism
raw_rdd = spark.sparkContext.wholeTextFiles(INPUT_PATH, minPartitions=20)
books_df = raw_rdd.toDF(["file_path", "text"])

# 3. ROBUST METADATA EXTRACTION
print("[2/5] Extracting Author & Year...")

# A. Extract Author (With Cleaning for "(1812-1870)" cases)
# We extract the raw line, then remove parenthetical text, then trim.
author_df = books_df.withColumn("raw_author", regexp_extract(col("text"), r"(?i)Author:\s*(.+)", 1)) \
    .withColumn("author", trim(regexp_replace(col("raw_author"), r"\(.*?\)", "")))

# B. Extract Release Date
author_df = author_df.withColumn("date_str", regexp_extract(col("text"), r"(?i)Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4}|[A-Za-z]+, \d{4})", 1))

# C. Robust Date Parsing
author_df = author_df.withColumn("year", coalesce(
    year(to_date(col("date_str"), "MMMM d, yyyy")),
    year(to_date(col("date_str"), "MMM d, yyyy")),
    year(to_date(col("date_str"), "MMMM, yyyy")),
    year(to_date(col("date_str"), "MMM, yyyy"))
).cast(IntegerType()))

# D. Filter Valid Data
# distinct() here means: If Mark Twain publishes 2 books in 1884, we keep only one entry for (Twain, 1884)
clean_df = author_df.filter((col("author") != "") & (col("year").isNotNull())) \
    .select("author", "year") \
    .distinct()

clean_df.cache()
author_year_count = clean_df.count()
print(f"      > Found {author_year_count} unique (Author, Year) active pairs.")

# 4. NETWORK CONSTRUCTION (SELF-JOIN)
print(f"[3/5] Building Influence Network (Window: {X_YEARS} years)...")

df_a = clean_df.alias("a")
df_b = clean_df.alias("b")

# Influence Condition: 
# 1. Different Author
# 2. Author A published BEFORE Author B
# 3. Author B published within X years of Author A
join_condition = (
    (col("a.author") != col("b.author")) & 
    (col("a.year") < col("b.year")) & 
    (col("b.year") <= col("a.year") + X_YEARS)
)

edges_df = df_a.join(df_b, join_condition) \
    .select(
        col("a.author").alias("influencer"),
        col("b.author").alias("follower"),
        col("a.year").alias("year_a"),
        col("b.year").alias("year_b")
    )

# 5. ANALYSIS: IN-DEGREE & OUT-DEGREE
print("[4/5] Calculating Network Metrics...")

# Out-Degree: Number of UNIQUE authors influenced
# Logic: If I influenced you in 1995 AND 1998, countDistinct ensures I only get +1 credit.
out_degree = edges_df.groupBy("influencer") \
    .agg(countDistinct("follower").alias("out_degree_count")) \
    .orderBy(desc("out_degree_count"))

# In-Degree: Number of UNIQUE authors who influenced me
in_degree = edges_df.groupBy("follower") \
    .agg(countDistinct("influencer").alias("in_degree_count")) \
    .orderBy(desc("in_degree_count"))

# 6. RESULTS
print("\n" + "="*60)
print(f"TOP 5 MOST INFLUENTIAL AUTHORS (Highest Out-Degree)")
print("="*60)
out_degree.show(5, truncate=False)

print("\n" + "="*60)
print(f"TOP 5 MOST INFLUENCED AUTHORS (Highest In-Degree)")
print("="*60)
in_degree.show(5, truncate=False)

print(f"Total Execution Time: {time.time() - start_time:.2f}s")
spark.stop()
