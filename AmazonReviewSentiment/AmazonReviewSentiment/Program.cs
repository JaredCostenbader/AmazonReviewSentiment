using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using SentimentAnalysis;
using static Microsoft.ML.DataOperationsCatalog;

namespace AmazonReviewAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create MLContext
            MLContext mlContext = new MLContext();

            // Load data
            TrainTestData splitDataView = LoadData(mlContext);

            // Build and train model
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            // Evaluate model
            Evaluate(mlContext, model, splitDataView.TestSet);

            // Use model with single item
            UseModelWithSingleItem(mlContext, model);

            // Use model with batch items
            UseModelWithBatchItems(mlContext, model);
        }

        static TrainTestData LoadData(MLContext mlContext)
        {
            // Path to the data file
            string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train-reviews-micro.csv");

            // Load data from text file
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: true, separatorChar: ',');

            // Split data into train and test sets
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }

        static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
        {
            // Convert the label column to Boolean type if it's Int32
            var dataProcessPipeline = mlContext.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean)  // Convert Label to Boolean
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.ReviewText)))  // Changed SentimentText to ReviewText
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

            // Train the model
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = dataProcessPipeline.Fit(trainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        static void Evaluate(MLContext mlContext, ITransformer model, IDataView testSet)
        {
            // Evaluate the model
            Console.WriteLine("=============== Evaluating Model accuracy with Test data ===============");
            IDataView predictions = model.Transform(testSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(SentimentData.Label));  // Changed Sentiment to Label
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            // Single test statement
            SentimentData sampleStatement = new SentimentData
            {
                ReviewText = "This was a very bad steak",  // Only initialize ReviewText once
                Title = ""  // Initialize Title once
            };

            // Make prediction
            var resultPrediction = predictionEngine.Predict(sampleStatement);

            // Output prediction
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            Console.WriteLine();
            Console.WriteLine($"Review: {resultPrediction.ReviewText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability}");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            // Batch test statements
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData { ReviewText = "This was a horrible meal", Title = "" },  // Only initialize ReviewText once
                new SentimentData { ReviewText = "I love this spaghetti.", Title = "" }  // Only initialize ReviewText once
            };

            // Load batch comments
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            // Make predictions
            IDataView predictions = model.Transform(batchComments);

            // Convert predictions to IEnumerable
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            // Output predictions
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Review: {prediction.ReviewText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability}");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
