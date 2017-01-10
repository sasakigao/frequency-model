import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.{SVMWithSGD, LogisticRegressionWithLBFGS, NaiveBayes}
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.configuration.{Strategy}
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.rdd.RDD

/**
 * 5 models altogther.
 * SVM, logistic regression, tree, random forests and naive bayes.
 */
object TrainEva {
	val svmMetricsPathHDFS = "hdfs:///netease/blend/metrics/svm/"
	val treeMetricsPathHDFS = "hdfs:///netease/blend/metrics/tree/"
	val logisticMetricsPathHDFS = "hdfs:///netease/blend/metrics/logistic/"
	val bayesMetricsPathHDFS = "hdfs:///netease/blend/metrics/bayes/"
	val forestMetricsPathHDFS = "hdfs:///netease/blend/metrics/forest/"

 	def trainAndEvaSVM(trainingRDD : RDD[LabeledPoint], testRDD : RDD[LabeledPoint], sc : SparkContext, timeStr : String) = {
  		val modelSVM = SVMWithSGD.train(trainingRDD, 100).clearThreshold()
		// modelSVM.setThreshold(0.8)		
		val scoresAndLabels = testRDD.map{ point =>
 	 		val score = modelSVM.predict(point.features)
  			(score, point.label)
		}
		binaryScoresMetricsOutput(scoresAndLabels, svmMetricsPathHDFS + timeStr, sc)
  	}	

  	def trainAndEvaLogistic(trainingRDD : RDD[LabeledPoint], testRDD : RDD[LabeledPoint], sc : SparkContext, timeStr : String) = {
  		val modelLogistic = new LogisticRegressionWithLBFGS()
  			.setNumClasses(2)
  			.run(trainingRDD)
  			.clearThreshold()
		val scoresAndLabels = testRDD.map{ point =>
 	 		val score = modelLogistic.predict(point.features)
  			(score, point.label)
		}
		binaryScoresMetricsOutput(scoresAndLabels, logisticMetricsPathHDFS + timeStr, sc)
  	}

  	def trainAndEvaTree(trainingRDD : RDD[LabeledPoint], testRDD : RDD[LabeledPoint], sc : SparkContext, timeStr : String) = {
  		val algo = Classification
		val impurity = Gini
		val maxDepth = 10
  		val numClasses = 2
		val maxBins = 32
		val categoricalFeaturesInfo = Map[Int, Int]()
		val strategy = new Strategy(algo, impurity, maxDepth)
	  	val modelTree = new DecisionTree(strategy)
	  		.run(trainingRDD)
		val predicationsAndLabels = testRDD.map{ point =>
 	 		val predication = modelTree.predict(point.features)
  			(predication, point.label)
		}
		binaryPredicationMetricsOutput(predicationsAndLabels, treeMetricsPathHDFS + timeStr, sc)
  	}
  	
  	def trainAndEvaForest(trainingRDD : RDD[LabeledPoint], testRDD : RDD[LabeledPoint], sc : SparkContext, timeStr : String) = {
  		val numClasses = 2
		val categoricalFeaturesInfo = Map[Int, Int]()
		val numTrees = 13
		val featureSubsetStrategy = "auto"
		val impurity = "gini"
		val maxDepth = 10
		val maxBins = 32
		val modelForest = RandomForest.trainClassifier(trainingRDD, numTrees, categoricalFeaturesInfo, 
			numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
		val predicationsAndLabels = testRDD.map{ point =>
 	 		val predication = modelForest.predict(point.features)
  			(predication, point.label)
		}
		binaryPredicationMetricsOutput(predicationsAndLabels, forestMetricsPathHDFS + timeStr, sc)
  	}

  	// Mind that features must be nonnegative.
  	def trainAndEvaBayes(trainingRDD : RDD[LabeledPoint], testRDD : RDD[LabeledPoint], sc : SparkContext, timeStr : String) = {
  		val modelBayes = new NaiveBayes()
  			.setLambda(1.0)
  			.setModelType("multinomial")
			.run(trainingRDD)
		val predicationsAndLabels = testRDD.map{ point =>
 	 		val prediction = modelBayes.predict(point.features)
  			(prediction, point.label)
		}
		binaryPredicationMetricsOutput(predicationsAndLabels, bayesMetricsPathHDFS + timeStr, sc)
  	}

  	def binaryScoresMetricsOutput(scoresAndLabels : RDD[(Double, Double)], metricsPath : String, sc : SparkContext) = {
  		val metrics = new BinaryClassificationMetrics(scoresAndLabels)
  		val auROC = metrics.areaUnderROC().toString
  		val auPR = metrics.areaUnderPR().toString
  		val fMeasure = metrics.fMeasureByThreshold().collect.mkString(" ")
  		val precision = metrics.precisionByThreshold().collect.mkString(" ")
  		val recall = metrics.recallByThreshold().collect.mkString(" ")
  		val metricsList = List(("AUC", auROC), ("AUPR", auPR), ("fMeasure", fMeasure), 
  			("Precision", precision), ("Recall", recall))  		
  		val metricsRDD = sc.parallelize(metricsList, 1)
  		metricsRDD.saveAsTextFile(metricsPath)
  	}

  	def binaryPredicationMetricsOutput(predicationsAndLabels : RDD[(Double, Double)], metricsPath : String, sc : SparkContext) = {
  		predicationsAndLabels.persist()
  		val accuracy = 1.0 * predicationsAndLabels.filter(x => x._1 == x._2).count / predicationsAndLabels.count
  		val precision = 1.0 * predicationsAndLabels.filter(x =>x._1 == x._2 && x._1 == 1.0).count / predicationsAndLabels.filter(_._1 == 1.0).count
  		val recall = 1.0 * predicationsAndLabels.filter(x =>x._1 == x._2 && x._1 == 1.0).count / predicationsAndLabels.filter(_._2 == 1.0).count
  		val f1Measure = 2 * precision * recall / (precision + recall)
  		val metrics = List(("accuracy", accuracy), ("precision", precision), ("recall", recall), ("f1Measure", f1Measure))
		sc.parallelize(metrics, 1).saveAsTextFile(metricsPath)
		predicationsAndLabels.unpersist(true)
  	}

}

