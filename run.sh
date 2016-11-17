#!/bin/bash

spark-submit \
    --master local \
    --deploy-mode client \
    --queue default \
    --driver-memory 512M \
    --executor-memory 512M \
    --num-executors 8 \
    --class SVMOnSample \
/home/sasaki/dev/netease/svm-sample/target/scala-2.10/svm-sample_2.10-0.1-SNAPSHOT.jar