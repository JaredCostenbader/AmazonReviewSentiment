using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace SentimentAnalysis
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public bool Label { get; set; }  // Changed Sentiment to Label

        [LoadColumn(1)]
        public string? Title { get; set; }

        [LoadColumn(2)]
        public string? ReviewText { get; set; } // Ensure this matches the input column name
    }

    public class SentimentPrediction : SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
