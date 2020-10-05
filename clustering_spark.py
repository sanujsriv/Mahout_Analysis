from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

dataset = spark.read.format("libsvm").load("/home/samira/Downloads/libsvm.data")
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)
predictions = model.transform(dataset)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
