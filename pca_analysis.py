
import os

import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType
)

from pyspark.ml.feature import (
    Imputer,
    VectorAssembler,
    StandardScaler,
    StringIndexer
)

from pyspark.ml.feature import PCA

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator
)

spark = (
    SparkSession.builder
    .appName("BreastCancerReferralPCA")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nSPARK SESSION CREATED SUCCESSFULLY")


try:

    schema = StructType([

        StructField("patient_id", StringType(), True),

        StructField("mean_radius", DoubleType(), True),
        StructField("mean_texture", DoubleType(), True),
        StructField("mean_perimeter", DoubleType(), True),
        StructField("mean_area", DoubleType(), True),
        StructField("mean_smoothness", DoubleType(), True),
        StructField("mean_compactness", DoubleType(), True),
        StructField("mean_concavity", DoubleType(), True),
        StructField("mean_concave_points", DoubleType(), True),
        StructField("mean_symmetry", DoubleType(), True),
        StructField("mean_fractal_dimension", DoubleType(), True),

        StructField("radius_error", DoubleType(), True),
        StructField("texture_error", DoubleType(), True),
        StructField("perimeter_error", DoubleType(), True),
        StructField("area_error", DoubleType(), True),
        StructField("smoothness_error", DoubleType(), True),
        StructField("compactness_error", DoubleType(), True),
        StructField("concavity_error", DoubleType(), True),
        StructField("concave_points_error", DoubleType(), True),
        StructField("symmetry_error", DoubleType(), True),
        StructField("fractal_dimension_error", DoubleType(), True),

        StructField("worst_radius", DoubleType(), True),
        StructField("worst_texture", DoubleType(), True),
        StructField("worst_perimeter", DoubleType(), True),
        StructField("worst_area", DoubleType(), True),
        StructField("worst_smoothness", DoubleType(), True),
        StructField("worst_compactness", DoubleType(), True),
        StructField("worst_concavity", DoubleType(), True),
        StructField("worst_concave_points", DoubleType(), True),
        StructField("worst_symmetry", DoubleType(), True),
        StructField("worst_fractal_dimension", DoubleType(), True),

        StructField("diagnosis", StringType(), True)

    ])


    file_path = "data/breast_cancer_referrals.csv"

    df = (
        spark.read
        .option("header", True)
        .schema(schema)
        .csv(file_path)
    )

    print("\nDATASET LOADED SUCCESSFULLY")

    print("\n" + "=" * 60)
    print("ORIGINAL DATASET INFORMATION")
    print("=" * 60)

    df.printSchema()

    original_count = df.count()

    print("\nTOTAL NUMBER OF RECORDS:", original_count)
    print("TOTAL NUMBER OF COLUMNS:", len(df.columns))

    print("\nFIRST FIVE RECORDS:")
    df.show(5, truncate=False)


    numeric_features = [

        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_smoothness",
        "mean_compactness",
        "mean_concavity",
        "mean_concave_points",
        "mean_symmetry",
        "mean_fractal_dimension",

        "radius_error",
        "texture_error",
        "perimeter_error",
        "area_error",
        "smoothness_error",
        "compactness_error",
        "concavity_error",
        "concave_points_error",
        "symmetry_error",
        "fractal_dimension_error",

        "worst_radius",
        "worst_texture",
        "worst_perimeter",
        "worst_area",
        "worst_smoothness",
        "worst_compactness",
        "worst_concavity",
        "worst_concave_points",
        "worst_symmetry",
        "worst_fractal_dimension"

    ]

    print("\n" + "=" * 60)
    print("DATA CLEANING: DUPLICATE REMOVAL")
    print("=" * 60)

    df = df.dropDuplicates()

    cleaned_count = df.count()

    print("Records before duplicate removal:", original_count)
    print("Records after duplicate removal:", cleaned_count)
    print("Duplicate records removed:", original_count - cleaned_count)


    print("\n" + "=" * 60)
    print("MISSING VALUES BEFORE IMPUTATION")
    print("=" * 60)

    missing_expressions = [

        F.sum(
            F.when(
                F.col(column).isNull() | F.isnan(F.col(column)),
                1
            ).otherwise(0)
        ).alias(column)

        for column in numeric_features

    ]

    missing_counts = df.agg(*missing_expressions).first().asDict()

    total_missing = 0

    for column, count in missing_counts.items():

        count = count or 0

        if count > 0:
            print(f"{column}: {count}")

        total_missing += count

    print("\nTOTAL MISSING NUMERICAL VALUES:", total_missing)


    print("\n" + "=" * 60)
    print("MEDIAN IMPUTATION")
    print("=" * 60)

    df = df.filter(
        F.col("diagnosis").isNotNull()
    )


    df = df.withColumn(
        "diagnosis",
        F.upper(F.trim(F.col("diagnosis")))
    )

    df = df.filter(
        F.col("diagnosis").isin("B", "M")
    )

    imputed_features = [
        column + "_imputed"
        for column in numeric_features
    ]

    imputer = Imputer(
        strategy="median",
        inputCols=numeric_features,
        outputCols=imputed_features
    )

    imputer_model = imputer.fit(df)

    imputed_df = imputer_model.transform(df)


    df = imputed_df.select(
        "patient_id",

        *[
            F.col(column + "_imputed").alias(column)
            for column in numeric_features
        ],

        "diagnosis"
    )

    print("MEDIAN IMPUTATION COMPLETED SUCCESSFULLY")


    print("\n" + "=" * 60)
    print("CLEANED DATASET RESULTS")
    print("=" * 60)

    final_count = df.count()

    print("FINAL NUMBER OF RECORDS:", final_count)
    print("FINAL NUMBER OF COLUMNS:", len(df.columns))

    print("\nCLEANED DATASET SCHEMA:")
    df.printSchema()

    print("\nFIRST FIVE CLEANED RECORDS:")
    df.show(5, truncate=False)

    remaining_missing = df.agg(
        *[
            F.sum(
                F.when(
                    F.col(column).isNull() | F.isnan(F.col(column)),
                    1
                ).otherwise(0)
            ).alias(column)

            for column in numeric_features
        ]
    ).first().asDict()

    remaining_total = sum(
        value or 0
        for value in remaining_missing.values()
    )

    print("\nRemaining missing numerical values:", remaining_total)

    if remaining_total == 0:
        print("SUCCESS: All numerical missing values have been handled.")
    else:
        raise ValueError("Missing numerical values remain after imputation.")


    print("\n" + "=" * 60)
    print("FEATURE ASSEMBLY")
    print("=" * 60)

    assembler = VectorAssembler(
        inputCols=numeric_features,
        outputCol="assembled_features",
        handleInvalid="error"
    )

    assembled_df = assembler.transform(df)

    print(
        "Successfully combined",
        len(numeric_features),
        "numerical features."
    )

    print("\nASSEMBLED FEATURE VECTORS:")

    assembled_df.select(
        "assembled_features"
    ).show(5, truncate=False)


    print("\n" + "=" * 60)
    print("FEATURE STANDARDIZATION")
    print("=" * 60)

    scaler = StandardScaler(
        inputCol="assembled_features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(assembled_df)

    scaled_df = scaler_model.transform(assembled_df)

    print("FEATURE STANDARDIZATION COMPLETED SUCCESSFULLY")

    print("\nSTANDARDIZED FEATURE VECTORS:")

    scaled_df.select(
        "scaled_features"
    ).show(5, truncate=False)


    print("\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 60)

    pca = PCA(
        k=2,
        inputCol="scaled_features",
        outputCol="pca_features"
    )

    pca_model = pca.fit(scaled_df)

    pca_results = pca_model.transform(scaled_df)

    print("PCA COMPLETED SUCCESSFULLY")
    print("Original numerical features:", len(numeric_features))
    print("Reduced principal components: 2")


    print("\n" + "=" * 60)
    print("EXPLAINED VARIANCE RESULTS")
    print("=" * 60)

    explained_variance = pca_model.explainedVariance.toArray()

    pc1_variance = explained_variance[0] * 100
    pc2_variance = explained_variance[1] * 100

    total_variance = pc1_variance + pc2_variance

    print(f"Principal Component 1: {pc1_variance:.2f}%")
    print(f"Principal Component 2: {pc2_variance:.2f}%")

    print(f"Total Variance Explained: {total_variance:.2f}%")


    print("\n" + "=" * 60)
    print("FIRST FIVE PCA RESULTS")
    print("=" * 60)

    pca_results.select(
        "patient_id",
        "diagnosis",
        "pca_features"
    ).show(5, truncate=False)


    print("\n" + "=" * 60)
    print("PCA SCATTER PLOT")
    print("=" * 60)

    plot_data = pca_results.select(
        "diagnosis",
        "pca_features"
    ).collect()

    pc1_b = []
    pc2_b = []

    pc1_m = []
    pc2_m = []

    for row in plot_data:

        pc1 = float(row["pca_features"][0])
        pc2 = float(row["pca_features"][1])

        if row["diagnosis"] == "B":

            pc1_b.append(pc1)
            pc2_b.append(pc2)

        elif row["diagnosis"] == "M":

            pc1_m.append(pc1)
            pc2_m.append(pc2)

    plt.figure(figsize=(10, 7))

    plt.scatter(
        pc1_b,
        pc2_b,
        label="Benign (B)",
        alpha=0.7
    )

    plt.scatter(
        pc1_m,
        pc2_m,
        label="Malignant (M)",
        alpha=0.7
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(
        "PCA Visualization of Breast Cancer Referrals"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    plot_path = "outputs/pca_scatter_plot.png"

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("PCA scatter plot saved to:", plot_path)

    plt.show()

    plt.close()


    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)


    indexer = StringIndexer(
        inputCol="diagnosis",
        outputCol="label",
        stringOrderType="alphabetAsc"
    )

    indexer_model = indexer.fit(pca_results)

    indexed_data = indexer_model.transform(pca_results)

    print("Diagnosis label mapping:")

    for index, label in enumerate(indexer_model.labels):
        print(f"{label} = {index}")

    # Retain PCA features and target label.

    model_data = indexed_data.select(
        "pca_features",
        "label"
    )


    training_data, testing_data = model_data.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    training_count = training_data.count()
    testing_count = testing_data.count()

    print("\nTraining records:", training_count)
    print("Testing records:", testing_count)

    if training_count == 0 or testing_count == 0:
        raise ValueError("Training or testing dataset is empty.")

    # Create logistic regression model.

    lr = LogisticRegression(
        featuresCol="pca_features",
        labelCol="label",
        maxIter=100
    )


    lr_model = lr.fit(training_data)

    print("\nLOGISTIC REGRESSION MODEL TRAINED SUCCESSFULLY")


    predictions = lr_model.transform(testing_data)

    print("\nSAMPLE PREDICTIONS:")

    predictions.select(
        "label",
        "prediction",
        "probability"
    ).show(10, truncate=False)


    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted Precision: {precision * 100:.2f}%")
    print(f"Weighted Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1_score * 100:.2f}%")

    print("\nCONFUSION MATRIX:")

    predictions.groupBy(
        "label",
        "prediction"
    ).count().orderBy(
        "label",
        "prediction"
    ).show()



    print("\n" + "=" * 60)
    print("FINAL PCA ANALYSIS SUMMARY")
    print("=" * 60)

    print("Original records:", original_count)
    print("Final records:", final_count)

    print("Duplicate records removed:", original_count - cleaned_count)

    print("Original numerical features:", len(numeric_features))

    print("Reduced principal components: 2")

    print(f"PC1 variance: {pc1_variance:.2f}%")

    print(f"PC2 variance: {pc2_variance:.2f}%")

    print(f"Total variance retained: {total_variance:.2f}%")

    print(f"Logistic regression accuracy: {accuracy * 100:.2f}%")

    print("\nPCA ANALYSIS AND LOGISTIC REGRESSION COMPLETED SUCCESSFULLY")

    print("Visualization:", plot_path)


finally:


    spark.stop()

    print("\nSPARK SESSION CLOSED")
