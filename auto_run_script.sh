#!/bin/bash

# --- CONFIGURATION ---
INPUT_PATH="/user/hadoop/gutenberg/input/200.txt"
BASE_OUTPUT="/user/hadoop/gutenberg/output_"
JAR="WordCount.jar"
CLASS="WordCount"

# Define Split Sizes (Focusing on small sizes for your 8.3 MB file)
# 512KB, 1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB
SIZES=(524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864)

# Prepare Results File with a Header
LOG_FILE="results_final.txt"
printf "%-15s %-10s %-15s\n" "Split_Size(B)" "Mappers" "Time(ms)" > $LOG_FILE
echo "------------------------------------------------" >> $LOG_FILE

echo "Starting Experiments..."
echo "------------------------------------------------"

# --- EXECUTION LOOP ---
for SIZE in "${SIZES[@]}"
do
    OUTPUT_DIR="${BASE_OUTPUT}${SIZE}"

    # 1. Clean up previous output
    hdfs dfs -rm -r $OUTPUT_DIR > /dev/null 2>&1

    echo -n "Running Split=${SIZE}..."

    # 2. Run Hadoop Job & Capture ALL Output
    # '2>&1' captures both standard output and error logs
    JOB_OUTPUT=$(hadoop jar $JAR $CLASS $INPUT_PATH $OUTPUT_DIR $SIZE 2>&1)

    # 3. Extract Execution Time
    # Looking for our custom tag "TIMING_RESULT="
    TIME=$(echo "$JOB_OUTPUT" | grep "TIMING_RESULT=" | cut -d'=' -f2)

    # 4. Extract Number of Mappers
    # Looking for the standard Hadoop counter "Launched map tasks=N"
    MAPPERS=$(echo "$JOB_OUTPUT" | grep "Launched map tasks=" | cut -d'=' -f2 | tr -d ' ')

    # 5. Handle Failures
    if [ -z "$TIME" ]; then
        TIME="FAILED"
        MAPPERS="-"
        echo " [FAILED]"
    else
        echo " [DONE] -> Mappers: $MAPPERS | Time: $TIME ms"
    fi

    # 6. Log to file nicely formatted
    printf "%-15s %-10s %-15s\n" "$SIZE" "$MAPPERS" "$TIME" >> $LOG_FILE
done

echo "------------------------------------------------"
echo "Experiments Completed. Results:"
cat $LOG_FILE
