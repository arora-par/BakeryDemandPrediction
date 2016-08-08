package edu.neu.bigdata.course;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class BakeryDemandPrediction {

	@SuppressWarnings("serial")
	public static void main(String[] args) {

		// Initialize spark context
		SparkConf sparkConf = new SparkConf()
				.setAppName("BakeryDemandPrediction");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		SQLContext sqlContext = new SQLContext(jsc);
		List<String> outputInfo = new ArrayList<String>();
		List<Double> saveToFile = new ArrayList<Double>();

		// Load and parse the data file.
		StructType customSchema = new StructType(new StructField[] {
				new StructField("week", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("salesdepotid", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("channelid", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("routeid", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("clientid", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("productid", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("salesunit", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("salespesos", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("returnsunit", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("returnspesos", DataTypes.DoubleType, true,
						Metadata.empty()),
				new StructField("demandunit", DataTypes.DoubleType, true,
						Metadata.empty()) });

		DataFrame df = sqlContext
				.read()
				.format("com.databricks.spark.csv")
				.schema(customSchema)
				.option("header", "true")
				.load("hdfs://ec2-52-41-138-171.us-west-2.compute.amazonaws.com:9000/user/root/input/train.csv");

		// Analyze the dataset

		// df.describe("week").show();
		// df.describe("week").toJavaRDD().saveAsTextFile("hdfs://ec2-52-41-138-171.us-west-2.compute.amazonaws.com:9000/user/root/output/week.txt");
		//
		//
		//
		// df.describe( "salesdepotid", "channelid", "routeid",
		// "salesunit", "salespesos", "returnsunit",
		// "returnspesos", "demandunit").show();
		// df.describe( "salesdepotid", "channelid", "routeid",
		// "salesunit", "salespesos", "returnsunit",
		// "returnspesos",
		// "demandunit").toJavaRDD().saveAsTextFile("hdfs://ec2-52-41-138-171.us-west-2.compute.amazonaws.com:9000/user/root/output/describeAll.txt");
		//

		// Split the data into training and test sets - week 3-7 for training
		// and week 8-9 for test

		DataFrame trainingData = df.where(df.col("week").leq(7.0));
		DataFrame testData = df.where(df.col("week").gt(7.0));

		// DATA TRANSFORMATION - TODO - try the means instead of regular productIds. Please note current code doesn't use this.
//		Map<String, String> demandAvg = new HashMap<String, String>();
//		demandAvg.put("demandunit", "avg");
//		DataFrame prodMeanDf = trainingData.groupBy(
//				trainingData.col("productid")).agg(demandAvg);
//
//		final Map<Double, Double> prodMeanMap = prodMeanDf.toJavaRDD()
//				.mapToPair(new PairFunction<Row, Double, Double>() {
//					@Override
//					public Tuple2<Double, Double> call(Row row)
//							throws Exception {
//						return new Tuple2<Double, Double>(row.getDouble(0), row
//								.getDouble(1));
//					}
//				}).collectAsMap();
//
//		JavaRDD<Row> dataWithMean = trainingData.toJavaRDD().map(
//				new Function<Row, Row>() {
//					@Override
//					public Row call(Row row) {
//
//						Row modifiedRow = RowFactory.create(row.get(0),
//								row.get(1), row.get(2), row.get(3), row.get(4),
//								prodMeanMap.get(row.getDouble(5)), row.get(6),
//								row.get(7), row.get(8), row.get(9), row.get(10));
//						return modifiedRow;
//					}
//
//				});

		// Adapt Training DataFrame to "label and features" notation
		// CONSIDER - Use VectorAssembler instead of manually adapting it
		JavaRDD<LabeledPoint> allDataLFRDD = df.toJavaRDD().map(
				new AdaptationFunction());
		DataFrame adaptedAllDF = sqlContext.createDataFrame(allDataLFRDD,
				LabeledPoint.class);

		JavaRDD<LabeledPoint> trainingDataLFRDD = trainingData.toJavaRDD().map(
				new AdaptationFunction());
		DataFrame adaptedTrainingDF = sqlContext.createDataFrame(
				trainingDataLFRDD, LabeledPoint.class);

		// Adapt Test DataFrame to "label and features" notation

		JavaRDD<LabeledPoint> testDataLFRDD = testData.toJavaRDD().map(
				new AdaptationFunction());
		DataFrame adaptedTestDF = sqlContext.createDataFrame(testDataLFRDD,
				LabeledPoint.class);



		// GBT REGRESSION
		VectorIndexerModel featureIndexer = new VectorIndexer()
				.setInputCol("features").setOutputCol("indexedFeatures")
				.setMaxCategories(12).fit(adaptedAllDF);

		// Train a GBT model.
		GBTRegressor gbt = new GBTRegressor().setLabelCol("label")
				.setFeaturesCol("indexedFeatures").setMaxIter(10);

		// Chain indexer and GBT in a Pipeline
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
				featureIndexer, gbt });

		// Train model. This also runs the indexer.
		PipelineModel pipelineGbtModel = pipeline.fit(adaptedTrainingDF);

		// Make predictions.
		DataFrame predictions = pipelineGbtModel.transform(adaptedTestDF);

		// Select example rows to display.
		predictions.select("prediction", "label", "features").sample(true, 0.1)
				.show(100);
		predictions
				.select("prediction", "label", "features")
				.sample(true, 0.1)
				.toJavaRDD()
				.saveAsTextFile(
						"hdfs://ec2-52-41-138-171.us-west-2.compute.amazonaws.com:9000/user/root/output/GBTSamplePredictions.txt");

		// Select (prediction, true label) and compute test error
		RegressionEvaluator evaluator = new RegressionEvaluator()
				.setLabelCol("label").setPredictionCol("prediction")
				.setMetricName("rmse");
		double rmseGBT = evaluator.evaluate(predictions);
		System.out
				.println("GBT :: Root Mean Squared Error (RMSE) on test data = "
						+ rmseGBT);
		saveToFile.add(rmseGBT);

		// Compute Log error
		Double testGBTRFMSLESqr = predictions.select("prediction", "label")
				.toJavaRDD().map(new Function<Row, Double>() {

					@Override
					public Double call(Row row) throws Exception {
						Double diff = Math.log(row.getDouble(0) + 1)
								- Math.log(row.getDouble(0) + 1);
						return diff * diff;
					}
				}).reduce(new Function2<Double, Double, Double>() {
					@Override
					public Double call(Double a, Double b) {
						return a + b;
					}
				})
				/ predictions.count();

		Double testGBTRFMSLE = Math.sqrt(testGBTRFMSLESqr);
		saveToFile.add(testGBTRFMSLE);

		JavaRDD<Double> testRFMSERDD = jsc.parallelize(saveToFile);
		testRFMSERDD
				.saveAsTextFile("hdfs://ec2-52-41-138-171.us-west-2.compute.amazonaws.com:9000/user/root/output/RMSE.txt");

		System.out.println("Test Mean Squared Logarithmic Error: "
				+ testRFMSERDD);

		GBTRegressionModel gbtModel = (GBTRegressionModel) (pipelineGbtModel
				.stages()[1]);
		System.out.println("Learned regression GBT model:\n"
				+ gbtModel.toDebugString());

		
	}

	private static final class AdaptationFunction implements
			Function<Row, LabeledPoint> {

		private static final long serialVersionUID = -7360100685567357892L;

		@Override
		public LabeledPoint call(Row row) {
			return new LabeledPoint(row.getDouble(row.size() - 1),
					Vectors.dense(getAllButLastAsPrimitiveDoubleArray(row)));
		}
	}

	private static double[] getAllButLastAsPrimitiveDoubleArray(Row row) {
		double[] primArr = new double[row.length() - 1];

		for (int i = 0; i < row.length() - 1; i++) {
			primArr[i] = row.getDouble(i);
		}
		return primArr;

	}

}