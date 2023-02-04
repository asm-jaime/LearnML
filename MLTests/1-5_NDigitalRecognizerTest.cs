using FluentAssertions;
using Keras.PreProcessing.Image;
using Numpy;
using NUnit.Framework;

namespace MLTests
{
    public interface IImageToVector
    {
        float[] GetVectorOnPath(string path);
    }

    public class ImageToVector : IImageToVector
    {
        private const string DefaultImageType = "grayscale";
        private static readonly (int, int) DefaultSize = (28, 28);

        public ImageToVector()
        {

        }
        float[] GetNormalizedVector(float[] vector)
        {
            var max = vector.Max();
            var min = vector.Min();
            for(var i = 0; i < vector.Length; i++) vector[i] /= max;
            return vector;
        }

        public float[] GetVectorOnPath(string path)
        {
            var image = ImageUtil.LoadImg(path, DefaultImageType, target_size: DefaultSize);
            NDarray imageArray = ImageUtil.ImageToArray(image);
            return GetNormalizedVector(imageArray.GetData<float>());
        }
    }

    public class ImageToVectorTest
    {
        [Test]
        public void LoadImageTest()
        {
            var projectPath = "D:/projects.active/LearnML";
            var path = Path.GetFullPath(@$"{projectPath}/data/0/1.jpg");
            IImageToVector imgToVector = new ImageToVector();
            var vector = imgToVector.GetVectorOnPath(path);
            vector.Sum().Should().NotBe(0.0f);
        }
    }

    public class NDigitalRecognizer
    {
        private ImageToVector _vectorReader;
        private string _trainDataPath;
        private float[][] _weights;
        private float _alpha;
        private const int DigitsLength = 10;
        private const float DefaultAlpha = 0.1f;

        public NDigitalRecognizer(ImageToVector vectorReader, int inputSize, int outputSize, string trainDataPath, float alpha = DefaultAlpha)
        {
            _alpha = alpha;

            _vectorReader = vectorReader;
            _trainDataPath = trainDataPath;

            _weights = new float[outputSize].Select(element => new float[inputSize]).ToArray();
        }

        private float MulSVV(float[] vecA, float[] vecB)
        {
            float result = 0;
            for (int i = 0; i < vecA.Length; i++)
            {
                result = result + vecA[i] * vecB[i];
            }
            return result;
        }

        private float[] MulVM(float[] vec, float[][] matrix)
        {
            var result = new float[matrix.Length];
            for (var index = 0; index < result.Length; ++index)
            {
                result[index] = MulSVV(vec, matrix[index]);
            }
            return result;
        }

        private float[] SubVE(float[] vec, float element)
        {
            var result = new float[vec.Length];
            for (int i = 0; i < vec.Length; i++)
            {
                result[i] = vec[i] - element;
            }
            return result;

        }

        private float[] MulVV(float[] vecA, float[] vecB)
        {
            var result = new float[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] * vecB[i];
            }
            return result;
        }

        private float[] SubVV(float[] vecA, float[] vecB)
        {
            var result = new float[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] - vecB[i];
            }
            return result;
        }


        private float[][] ProdVV(float[] vecA, float[] vecB, float alpha)
        {
            float[][] result = vecA.Select(element => new float[vecB.Length]).ToArray();
            for(int row = 0; row < vecA.Length; row++)
            {
                for(int col = 0; col < vecB.Length; col++)
                {
                    result[row][col] = vecA[row] * vecB[col] * alpha;
                }
            }
            return result;
        }

        private float[][] GradientLearn(float[] input, float[][] weights, float[] predictionGoal, float alpha)
        {
            /*
                var prediction = input.dot(resultWeights);
                var delta = prediction - predictionGoal;
                resultWeights = resultWeights - alpha * (input * delta);
            */

            var result = weights.Select(row => row.ToArray()).ToArray();
            var prediction = MulVM(input, result);
            var deltas = SubVV(prediction, predictionGoal);
            var weightsDeltas = ProdVV(deltas, input, alpha);

            for(var row = 0; row < deltas.Length; ++row)
            {
                result[row] = SubVV(result[row], weightsDeltas[row]);
            }

            return result;
        }

        private float[] GetPredictionGoalVector(int digit)
        {
            var result = new float[DigitsLength];
            return result.Select((e, index) => digit.Equals(index) ? 1.0f : 0.0f).ToArray();
        }

        public void Learn(int iterations)
        {
            for(int iteration = 1; iteration <= iterations; iteration++)
            {
                for (var digit = 0; digit < DigitsLength; ++digit)
                {
                    var predictionGoal = GetPredictionGoalVector(digit);
                    var dataPath = Path.Combine(_trainDataPath, $"{digit}", $"{iteration}.jpg");
                    var input = _vectorReader.GetVectorOnPath(dataPath);
                    _weights = GradientLearn(input, _weights, predictionGoal, _alpha);
                }
            }
        }

        public (int, float) Predict(string dataPath)
        {
            var input = _vectorReader.GetVectorOnPath(dataPath);
            var resultVector =  MulVM(input, _weights);
            var result = resultVector
                .Select((e, i) => new { Element = e, Index = i })
                .OrderBy(element => element.Element).Last();
            return (result.Index, result.Element);
        }
    }

    public class NDigitalRecognizerTest
    {
        [Test]
        public void ShouldRecognizeDigit()
        {
            var projectPath = "D:/projects.active/LearnML";
            var trainDataPath = Path.GetFullPath(@$"{projectPath}/data");
            var digitZeroPath = Path.Combine(trainDataPath, "0", "155.jpg");
            var digitOnePath = Path.Combine(trainDataPath, "1", "1055.jpg");
            var digitTwoPath = Path.Combine(trainDataPath, "2", "1055.jpg");
            var digitThreePath = Path.Combine(trainDataPath, "3", "1055.jpg");
            var digitFourPath = Path.Combine(trainDataPath, "4", "1055.jpg");
            var digitFivePath = Path.Combine(trainDataPath, "5", "55.jpg");
            var digitSixPath = Path.Combine(trainDataPath, "6", "1055.jpg");
            var digitSevenPath = Path.Combine(trainDataPath, "7", "1055.jpg");
            var digitEightPath = Path.Combine(trainDataPath, "8", "1055.jpg");
            var digitNinePath = Path.Combine(trainDataPath, "9", "55.jpg");

            var inputSize = 28 * 28;
            var outputSize = 10;

            var alpha = 0.001f;
            var iterations = 100;

            var recognizer = new NDigitalRecognizer(new ImageToVector(), inputSize, outputSize, trainDataPath, alpha);

            recognizer.Learn(iterations);

            {
                var (digit, value) = recognizer.Predict(digitNinePath);
                Assert.That(digit, Is.EqualTo(9));
                //Assert.That(value, Is.AtLeast(0.5f));
            }
            {
                var (digit, value) = recognizer.Predict(digitFivePath);
                Assert.That(digit, Is.EqualTo(5));
                //Assert.That(value, Is.AtLeast(0.5f));
            }

            //recognizer.Predict(digitOnePath)[zero].Should().BeInRange(0.0f, 0.6f);
            //recognizer.Predict(digitZeroPath)[zero].Should().BeInRange(0.8f, 1.2f);
        }
    }
}
