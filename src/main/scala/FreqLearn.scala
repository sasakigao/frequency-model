import scala.collection.mutable._
import scala.collection.Iterable

import org.apache.spark.{SparkConf, SparkContext, HashPartitioner}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD

import java.text.SimpleDateFormat


object FreqLearn {
	val appName = "Frequency-based test"

    	// Read from
	val playerLogHDFS = "hdfs:///netease/datum/labelled/"
	val operationByUserFile = "hdfs:///netease/datum/operations-user"
	val pluginPlayerFile = "hdfs:///netease/datum/waiguaid"

	// Write to
	val totalPlayersHDFS = "hdfs:///netease/blend/players/total/"
	val validPlayersHDFS = "hdfs:///netease/blend/players/valid/"
	val validDurationsHDFS= "hdfs:///netease/blend/players/valid-duration/"
	val positiveValidDurationsHDFS= "hdfs:///netease/blend/players/valid-duration-positive/"
	val labelFeaturesHDFS = "hdfs:///netease/blend/features/"

	val Postive = 0.0
	val Negative = 1.0

	def main(args: Array[String]) = {
	  	val confSpark = new SparkConf().setAppName(appName)
	  	confSpark.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  		val sc = new SparkContext(confSpark)

  		// Load 2 lookup files and broadcast them
  		// opeLookup is like (operation, index of userid), pluginLookup is like (date, userid)
		val opeLookup = sc.textFile(operationByUserFile)
			.collect
			.map{x => val f = x.split(","); (f(0), f(1))}
			.toMap
		val pluginLookup = sc.textFile(pluginPlayerFile)
			.map{x => val f = x.split(","); (f(0), f(1))}
			.groupByKey
			.collect
			.toMap
			.map(x => (x._1, x._2.toArray))
		val opeLookupBc = sc.broadcast(opeLookup)
		val pluginLookupBc = sc.broadcast(pluginLookup)

		val rawDatum = sc.textFile(playerLogHDFS)
		
		// Select those in the operation list and valid of three fields, then
		// grouped by (player, date), value comes to grouped (operation, timestamp)
		val playerRefinedRDD = rawDatum.map{ line =>
			val opeId = opeXtract(line)
			val timestamp = timeXtract(line)
			val playerId = playerXtract(line, opeId, opeLookupBc.value)
			(playerId, (opeId, timestamp))
		}.filter(x => x._1 != None && x._2._1 != None && x._2._2 != None)
		.map{ case (x, (a, b)) => ((x.get, b.get.split(" ")(0)), (a.get, toMillis(b.get)))}
		.groupByKey
		playerRefinedRDD.persist()
		Validation.totalPlayersCollect(playerRefinedRDD, totalPlayersHDFS, false)

		// Players whose login times equal to logout
		val validPlayersRDD = Validation.validPlayersCollect(playerRefinedRDD, validPlayersHDFS, false)
		// Count the online duration from validPlayersRDD
		val validPlayersDurationRDD = Validation.validDurationsCollect(validPlayersRDD, validDurationsHDFS, false)
		// Remove those whose durations are negative
		val positiveValidPlayersDurationRDD = Validation.validPositive(validPlayersDurationRDD, positiveValidDurationsHDFS, false)
		// Use duration-valid players to construct freq-based features
		val actFreqRDD = positiveValidPlayersDurationRDD.map{ case (playerTime, (duration, optArray)) =>
			(playerTime, freqByActs(duration, optArray, opeLookupBc.value))
		}

		val labelledActFreqRDD = actFreqRDD.map{ case (playerTime, actResults) =>
			if ((pluginLookupBc.value)(playerTime._2) contains playerTime._1) LabeledPoint(1.0, Vectors.dense(actResults))
				else LabeledPoint(0.0, Vectors.dense(actResults))
		}
		
		playerRefinedRDD.unpersist(true)
		validPlayersRDD.unpersist(true)
		validPlayersDurationRDD.unpersist(true)
		positiveValidPlayersDurationRDD.unpersist(true)

		// Save the labelled features in MLLib format
		labelledActFreqRDD.persist()
		// MLUtils.saveAsLibSVMFile(labelledActFreqRDD.repartition(5), labelFeaturesHDFS)

		// Training part. Features normalized. 
		// val labelledActFreqRDD = MLUtils.loadLibSVMFile(sc, labelFeaturesHDFS, 200)
		val normalizer = new Normalizer()
		val normalizedLabelledActFreqRDD = labelledActFreqRDD.map(x => LabeledPoint(x.label, normalizer.transform(x.features)))
  		val splits = normalizedLabelledActFreqRDD.randomSplit(Array(0.8, 0.2), seed = 22L)
		val trainingRDD = splits(0)
		val testRDD = splits(1)
		trainingRDD.persist()
		testRDD.persist()
		println(s"sasaki -train ${trainingRDD.filter(_.label == 0).count} ${trainingRDD.filter(_.label == 1.0).count}")
		println(s"sasaki -test ${testRDD.filter(_.label == 0).count} ${testRDD.filter(_.label == 1.0).count}")

		TrainEva.trainAndEvaSVM(trainingRDD, testRDD, sc)
		TrainEva.trainAndEvaLogistic(trainingRDD, testRDD, sc)
		TrainEva.trainAndEvaTree(trainingRDD, testRDD, sc)
		TrainEva.trainAndEvaBayes(trainingRDD, testRDD, sc)
		
		labelledActFreqRDD.unpersist(true)
		trainingRDD.unpersist(true)
		testRDD.unpersist(true)
    		sc.stop()
  	}

	// Compute the frequency of each action
  	def freqByActs(duration : Double, optArray : Array[String], opeLookup : collection.immutable.Map[String, String]) = {
  		val opeMap = opeLookup.keys.zipWithIndex.toMap
  		val counterInit = (0 until opeLookup.size).toBuffer.map{x : Int => 0}
  		optArray.foreach{ oneOpe =>
  			val index = opeMap(oneOpe)             // playerXtract has already removed actions out of opeLookup
  			counterInit(index) += 1
  		}
  		counterInit.map(_ / duration).toArray
  	}

  	// For a pure function consideration, Xtract returns the final Some[T] or None
  	def opeXtract(logLine : String) = {
  		val pattern = """(PLAYER|LOGIN)\|\[[0-9]{7}""".r             // PLAYER or LOGIN
  		val matchRes = pattern.findFirstIn(logLine)
  		if (matchRes != None) Some(matchRes.get.split('[').last) else None
  	}

  	def timeXtract(logLine : String) = {
  		val pattern = """^2016-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}""".r
  		val matchRes = pattern.findFirstIn(logLine)
  		if (matchRes != None) Some(matchRes.get.split('[').last) else None
  	}
  	// Also judge whether the operation is targetted, if not return None
  	def playerXtract(logLine : String, opeId : Option[String], 
  			opeLookup : collection.immutable.Map[String, String]) : Option[String] = {
  		val pattern = """(PLAYER|LOGIN)\|\[[0-9]{7}\].*""".r
  		val args = pattern.findFirstIn(logLine)
  		if (opeId != None && args != None) {
  			val index = opeLookup.get(opeId.get.split('[').last)
  			if (index != None) 
  				Some(args.get.split(']').last.split(',')(index.get.toInt).trim)
  			else 
  				None
  		} else {
  			None
  		}
  	}

  	def toMillis(timestamp : String) = {
  		val sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
  		sdf.parse(timestamp).getTime
  	}

}
