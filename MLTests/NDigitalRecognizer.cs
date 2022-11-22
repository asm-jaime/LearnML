namespace MLTests
{
    public class NDigitalRecognizer
    {
        private ImageToVector _vectorReader;
        private string _trainDataPath;
        private float[][] _weights;
        private int _sampleSize;
        private int _iterations;
        private float _alpha;
        private const int DigitsLength = 10;
        private const float DefaultAlpha = 0.1f;
        private const int DefaultIterations = 10;
        private const int DefaultSampleSize = 100;

        public NDigitalRecognizer(ImageToVector vectorReader, int inputSize, int outputSize, string trainDataPath, int sampleSize = DefaultSampleSize, float alpha = DefaultAlpha, int iterations = DefaultIterations)
        {
            _sampleSize = sampleSize;
            _iterations = iterations;
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

        private float[][] GradientLearn(float[] input, float[][] weights, float[] predictionGoal, float alpha, int iterations)
        {
            var result = weights.Select(row => row.ToArray()).ToArray();
            for (int i = 0; i < iterations; i++)
            {
                var prediction = MulVM(input, result);
                var deltas = SubVV(prediction, predictionGoal);
                var derivatives = MulVV(deltas, input);
                for (var row = 0; row < result.Length; ++row)
                {
                    result[row] = SubVE(result[row], derivatives[row] * alpha);
                }
            }
            return result;
        }

        private float[] GetPredictionGoalVector(int digit)
        {
            var result = new float[DigitsLength];
            return result.Select((e, index) => digit.Equals(index) ? 1.0f : 0.0f).ToArray();
        }

        public void Learn(int digit)
        {
            for (var sample = 1; sample <= _sampleSize; ++sample)
            {
                var predictionGoal = GetPredictionGoalVector(digit);
                var dataPath = Path.Combine(_trainDataPath, $"{digit}", $"{sample}.jpg");
                var input = _vectorReader.GetVectorOnPath(dataPath);
                _weights = GradientLearn(input, _weights, predictionGoal, _alpha, _iterations);
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
}
