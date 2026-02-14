import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {
  private final static IntWritable one = new IntWritable(1);
  private Text word = new Text();
  private IntWritable result = new IntWritable();
  
  public static class TokenizerMapper
       extends Mapper<Object, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
    
    // 1. Convert Hadoop Text object to a standard Java String
    String line = value.toString();

    // 2. Remove Punctuation
    // The regex "[^a-zA-Z0-9\s]" means: 
    // "Find any character that is NOT (^) a letter (a-z, A-Z), a number (0-9), or a whitespace (\s)"
    // Replace those characters with an empty string ""
    line = line.replaceAll("[^a-zA-Z0-9\\s]", "");

    // 3. Tokenize the clean line into words
    StringTokenizer itr = new StringTokenizer(line);

    // 4. Iterate through words and write them to output
    while (itr.hasMoreTokens()) {
        word.set(itr.nextToken()); // Put the string into the Hadoop Text object
        context.write(word, one);  // Emit (word, 1)
    }
}
  }

  public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
    
    int sum = 0;
    
    // 1. Iterate through the list of values (e.g., [1, 1, 1, 1])
    for (IntWritable val : values) {
        sum += val.get(); // .get() converts IntWritable to Java int
    }
    
    // 2. Wrap the sum back into a Hadoop Object
    result.set(sum);
    
    // 3. Write the final result (Word, TotalCount)
    context.write(key, result);
}
  }

  public static void main(String[] args) throws Exception {
    long startTime = System.currentTimeMillis();
    Configuration conf = new Configuration();

    // --- NEW LOGIC START ---
    // If a 3rd argument is provided (args[2]), use it as the split size.
    if (args.length > 2) {
        long splitSize = Long.parseLong(args[2]);
        conf.setLong("mapreduce.input.fileinputformat.split.maxsize", splitSize);
        System.out.println(">>> Setting Split Size to: " + splitSize);
    }
    // --- NEW LOGIC END ---

    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(WordCount.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    
    boolean success = job.waitForCompletion(true);

    long endTime = System.currentTimeMillis();
    System.out.println("TIMING_RESULT=" + (endTime - startTime)); // Simple tag for our script to grep
    System.exit(success ? 0 : 1);
  }
}
