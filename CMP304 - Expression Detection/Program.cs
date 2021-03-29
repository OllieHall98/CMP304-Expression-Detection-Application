using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.IO;
using System.Threading;
using System.Drawing;
using System.Media;


namespace CMP304___Expression_Detection
{
    public struct Vector2
    {
        public float x;
        public float y;

        public Vector2(float x, float y)
        {
            this.x = x;
            this.y = y;
        }
    }

    public enum Datatype { Training, Testing };

    public class ExpressionDetectionSystem
    {
        // file paths
        private static string currentFilePath = "";
        private static string trainingOutput = @"training_feature_vectors.csv";
        private static string testingOutput = @"testing_feature_vectors.csv";

        public static Datatype currentDataType = new Datatype();

        private static MLContext mlContext;

        public static Form1 form;

        [STAThread]

        //The main program entry point
        static void Main(string[] args)
        {
            Application.SetHighDpiMode(HighDpiMode.SystemAware);
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }

        public static void TrainModel()
        {
            mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromTextFile<FeatureInputData>(trainingOutput, hasHeader: true, separatorChar: ',');

            var trainTestData = RearrangeData(dataView);

            var featureVectorName = "Features";
            var labelColumnName = "Label";
            var pipeline = mlContext
                .Transforms.Conversion.MapValueToKey(inputColumnName: nameof(FeatureInputData.label), outputColumnName: labelColumnName).
                Append(mlContext.Transforms.Concatenate(featureVectorName,
                nameof(FeatureInputData.leftEyebrow),
                nameof(FeatureInputData.rightEyebrow),
                nameof(FeatureInputData.leftLip),
                nameof(FeatureInputData.rightLip),
                nameof(FeatureInputData.lipWidth),
                nameof(FeatureInputData.lipHeight)))
                .Append(mlContext.Transforms.NormalizeMinMax(featureVectorName, featureVectorName))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName, featureVectorName))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainTestData.TrainSet);
            
            using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write)) { mlContext.Model.Save(model, dataView.Schema, fileStream); } 

            GenerateMetrics(trainTestData.TestSet);
        }

        static void GenerateMetrics(IDataView dataView)
        {
            DataViewSchema modelSchema;
            ITransformer model = mlContext.Model.Load("model.zip", out modelSchema);

            var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(dataView));

            string metrics = ($"* Metrics for Multi-class Classification model - Test Data\n") +
                                 ($"* MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}\n") +
                                 ($"* MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}\n") +
                                 ($"* LogLoss:          {testMetrics.LogLoss:#.###}\n") +
                                 ($"* LogLossReduction: {testMetrics.LogLossReduction:#.###}\n") +
                                 ($"* {testMetrics.ConfusionMatrix.GetFormattedConfusionTable():#.###}");

            System.IO.File.WriteAllText("MetricData.txt", metrics);

            Form1.DisplayMetrics(metrics);
        }

       
        static private DataOperationsCatalog.TrainTestData RearrangeData(IDataView testDataView)
        {
            //IDataView shuffledData = mlContext.Data.ShuffleRows(testDataView);

            // split the data 
            return mlContext.Data.TrainTestSplit(testDataView, testFraction: 0.2);
        }

        public static void SelectSet(string filepath, Datatype datatype)
        {
            currentFilePath = filepath;
            currentDataType = datatype;

            Thread thread = new Thread(new ThreadStart(CreateFeatureVectors));
            thread.Start();
        }

        public static void CreateFeatureVectors()
        {
            int faceCount = 0;
            float leftEyebrow, rightEyebrow, leftLip, rightLip, lipHeight, lipWidth;
            string output;

            if (currentDataType == Datatype.Testing)
                output = testingOutput;
            else 
                output = trainingOutput;

            string[] dirs = Directory.GetFiles(currentFilePath, "*.*", SearchOption.AllDirectories);

            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                string header = "leftEyebrow,rightEyebrow,leftLip,rightLip,lipWidth,lipHeight,label\n";

                // Create the CSV file and fill in the first line with the header
                System.IO.File.WriteAllText(output, header);

                foreach (string dir in dirs)
                {
                    // call function that sets the label based on what the filename contains
                    string label = DetermineLabel(dir);

                    // load input image
                    if (!(dir.EndsWith("png") || dir.EndsWith("jpg")))
                        continue;

                    var img = Dlib.LoadImage<RgbPixel>(dir);
                    
                    // find all faces in the image
                    var faces = fd.Operator(img);

                    // for each face draw over the facial landmarks
                    foreach (var face in faces)
                    {
                        // Write to the console displaying the progress and current emotion
                        Form1.SetProgress(faceCount, dirs.Length - 1);

                        // find the landmark points for this face
                        var shape = sp.Detect(img, face);

                        for (var i = 0; i < shape.Parts; i++)
                        {
                            RgbPixel colour = new RgbPixel(255, 255, 255);
                            var point = shape.GetPart((uint)i);
                            var rect = new DlibDotNet.Rectangle(point);
                            Dlib.DrawRectangle(img, rect, color: colour, thickness: 2);
                        }

                        SetFormImage(img);

                        leftEyebrow = CalculateLeftEyebrow(shape);
                        rightEyebrow = CalculateRightEyebrow(shape);
                        leftLip = CalculateLeftLip(shape);
                        rightLip = CalculateRightLip(shape);
                        lipWidth = CalculateLipWidth(shape);
                        lipHeight = CalculateLipHeight(shape);

                        using (System.IO.StreamWriter file = new System.IO.StreamWriter(output, true))
                        {
                            file.WriteLine(leftEyebrow + "," + rightEyebrow + "," + leftLip + "," + rightLip + "," + lipWidth + "," + lipHeight + "," + label);
                        }

                        // Increment count used for console output
                        faceCount++;
                    }
                }

                if (currentDataType == Datatype.Testing)
                {
                    var testDataView = mlContext.Data.LoadFromTextFile<FeatureInputData>(output, hasHeader: true, separatorChar: ',');
                    GenerateMetrics(testDataView);
                }

                Form1.HideImage();
            }
        }

        public static void SetFormImage(Array2D<RgbPixel> img)
        {
            Dlib.ResizeImage(img, 0.4);

            Bitmap image = new Bitmap(img.Columns, img.Rows, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            for(int i = 0; i < img.Columns; i++)
                for(int j = 0; j < img.Rows; j++)
                {
                    //Color c = Color.FromArgb(img[i][j].Red, img[i][j].Green, img[i][j].Blue);
                    //image.SetPixel(i, j, c);
                }

            image.RotateFlip(RotateFlipType.Rotate90FlipNone);

            Form1.SetImage(image);
        }

        public static string TestCustomImage(string dir)
        {
            DataViewSchema predictionPipelineSchema;
            ITransformer predictionPipeline = mlContext.Model.Load("model.zip", out predictionPipelineSchema);
            PredictionEngine<FeatureInputData, ExpressionPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<FeatureInputData, ExpressionPrediction>(predictionPipeline);
            var img = Dlib.LoadImage<RgbPixel>(dir);

            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                // find all faces in the image
                var faces = fd.Operator(img);

                // for each face draw over the facial landmarks
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    FeatureInputData inputData = new FeatureInputData
                    {
                        leftEyebrow    = CalculateLeftEyebrow(shape),
                        rightEyebrow   = CalculateRightEyebrow(shape),
                        leftLip        = CalculateLeftLip(shape),
                        rightLip       = CalculateRightLip(shape),
                        lipWidth       = CalculateLipWidth(shape),
                        lipHeight      = CalculateLipHeight(shape)
                    };

                    ExpressionPrediction prediction = predictionEngine.Predict(inputData);

                    return prediction.expression.ToString();

                }
            }
            return "N/A";
        }



        static string DetermineLabel(string dir)
        {
            if (dir.Contains("neutral")) return "neutral";
            if(dir.Contains("surprise")) return "surprise";
            if (dir.Contains("sadness")) return "sadness";
            if (dir.Contains("fear")) return "fear";
            if (dir.Contains("disgust")) return "disgust";
            if (dir.Contains("anger")) return "anger";
            if (dir.Contains("joy")) return "joy";
            else return "N/A";
        }

        static float CalculateLeftEyebrow(FullObjectDetection shape)
        {
            float result = 0;
            float NormalisationDistance = CalculateDistance(shape, 21, 39);

            for (uint i = 18; i <= 21; i++) result += CalculateDistance(shape, i, 39) / NormalisationDistance;

            return result;
        }

        static float CalculateRightEyebrow(FullObjectDetection shape)
        {
            float result = 0;
            float NormalisationDistance = CalculateDistance(shape, 22, 42);

            for (uint i = 22; i <= 25; i++) result += CalculateDistance(shape, i, 42) / NormalisationDistance;

            return result;
        }

        static float CalculateLeftLip(FullObjectDetection shape)
        {
            float result = 0;
            float NormalisationDistance = CalculateDistance(shape, 33, 51);

            for (uint i = 48; i <= 50; i++) result += CalculateDistance(shape, i, 33) / NormalisationDistance;

            return result;
        }

        static float CalculateRightLip(FullObjectDetection shape)
        {
            float result = 0;
            float NormalisationDistance = CalculateDistance(shape, 33, 51);

            for (uint i = 52; i <= 54; i++) result += CalculateDistance(shape, i, 33) / NormalisationDistance;

            return result;
        }

        static float CalculateLipWidth(FullObjectDetection shape)
        {
            return CalculateDistance(shape, 48, 54) / CalculateDistance(shape, 33, 51);
        }

        static float CalculateLipHeight(FullObjectDetection shape)
        {
            return CalculateDistance(shape, 51, 57) / CalculateDistance(shape, 33, 51);
        }

        static float CalculateDistance(FullObjectDetection shape, uint point1, uint point2)
        {
            return MathF.Sqrt(MathF.Pow(shape.GetPart(point1).X - shape.GetPart(point2).X, 2) + MathF.Pow(shape.GetPart(point1).Y - shape.GetPart(point2).Y, 2));
        }
        }
    }


    public class FeatureInputData
    {
        [LoadColumn(0)]
        public float leftEyebrow { get; set; }

        [LoadColumn(1)]
        public float rightEyebrow { get; set; }

        [LoadColumn(2)]
        public float leftLip { get; set; }

        [LoadColumn(3)]
        public float rightLip { get; set; }

        [LoadColumn(4)]
        public float lipWidth { get; set; }

        [LoadColumn(5)]
        public float lipHeight { get; set; }

        [LoadColumn(6)]
        public string label { get; set; }
    }

    class ExpressionPrediction
    {
        [ColumnName("PredictedLabel")]
        public string expression { get; set; }
    }

