import scala.collection.Iterable

import org.apache.spark.rdd.RDD


object Validation {
	val LOGIN = "5300030"
	val LOGOUT = "5300035"	

	// Collect the total players
	def totalPlayersCollect(playerRefinedRDD : RDD[((String, String), Iterable[(String, Long)])], 
			totalPlayersHDFS : String, toStore : Boolean) : Unit = {
		if (toStore) {
			val playersRDD = playerRefinedRDD.keys
			playersRDD.repartition(5).saveAsTextFile(totalPlayersHDFS)	
		}
	}

	// Collect the valid players on durations
	def validPlayersCollect(playerRefinedRDD : RDD[((String, String), Iterable[(String, Long)])], 
			validPlayersHDFS : String, toStore : Boolean) = {
		val validPlayersRDD = playerRefinedRDD.filter{ case (_, logs) =>
			val logsArray = logs.toArray
			val logins = logsArray.filter(_._1 == LOGIN)
			val logouts = logsArray.filter(_._1 == LOGOUT)
			logins.size == logouts.size
		}
		validPlayersRDD.persist()
		if (toStore) {
			val validRDD = validPlayersRDD.keys
			validRDD.repartition(5).saveAsTextFile(validPlayersHDFS)
		}
		validPlayersRDD
	}

	// Compute the valid durations
	def validDurationsCollect(validPlayersRDD : RDD[((String, String), Iterable[(String, Long)])], 
			validDurationsHDFS : String, toStore : Boolean) = {
		val validDurationsCountsRDD = validPlayersRDD.map{ case (playerDate, logs) =>
			val logsArray = logs.toArray
			val logins = logsArray.filter(_._1 == LOGIN).map(_._2).sum
			val logouts = logsArray.filter(_._1 == LOGOUT).map(_._2).sum
			val duration = 1.0 * (logouts - logins) / (1000 * 60 * 60)
			(playerDate, (duration, logsArray.map(_._1)))
		}
		validDurationsCountsRDD.persist()
		if (toStore) {
			validDurationsCountsRDD
				.map(x => (x._1, x._2._1))
				.sortBy(x => x._2, false)
				.repartition(5)
				.saveAsTextFile(validDurationsHDFS)
		}
		validDurationsCountsRDD
	}

	def validPositive(validPlayersDurationRDD : RDD[((String, String), (Double, Array[String]))], 
			positiveValidDurationsHDFS : String, toStore : Boolean) = {
		val positiveValidPlayersDurationRDD = validPlayersDurationRDD.filter(_._2._1 > 0)
		positiveValidPlayersDurationRDD.persist()
		if (toStore) {
			positiveValidPlayersDurationRDD
				.map(x => (x._1, x._2._1))
				.sortBy(x => x._2, false)
				.repartition(5)
				.saveAsTextFile(positiveValidDurationsHDFS)
		}
		positiveValidPlayersDurationRDD
	}
	
}