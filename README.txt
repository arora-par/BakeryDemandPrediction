The project needs to be built using maven:

mvn clean compile package

The packaged jar is available under target directory.

The jar needs to be copied over to the cluster running the application.

Before running the application, make sure to copy the input data file (train.csv) in the 'input' directory of HDFS.

To run the application, the jar needs to be submitted to the spark runtime using the spark-submit command:

spark/bin/spark-submit --class edu.neu.bigdata.course.BakeryDemandPrediction --master spark://ec2-52-41-138-171.us-west-2.compute.amazonaws.com:7077 --packages com.databricks:spark-csv_2.11:1.4.0 bakery-0.0.1-SNAPSHOT.jar

The job runs for more than 1 hour on a cluster of 8 slaves each having 8GB memory.

The results of the spark job are available on the HDFS 'output' directory as well as printed on the console.