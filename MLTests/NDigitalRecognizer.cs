namespace MLTests
{
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
            //var derivatives = MulVV(input, deltas);
            for (var row = 0; row < result.Length; ++row)
            {
                result[row] = SubVE(result[row], derivatives[row] * alpha);
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
}
