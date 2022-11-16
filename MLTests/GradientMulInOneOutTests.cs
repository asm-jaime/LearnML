using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class GradientMulInOneOutTests
    {
        private double WeightsSum(double[] vecA, double[] vecB)
        {
            double result = 0;
            for (int i = 0; i < vecA.Length; i++)
            {
                result = result + vecA[i] * vecB[i];
            }
            return result;
        }

        private double[] WeightsSub(double[] vecA, double sub)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] - sub;
            }
            return result;

        }

        private double[] WeightsMul(double[] vecA, double[] vecB)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] * vecB[i];
            }
            return result;
        }

        private double[] GetLearnByGradientMul(double[] weights, double[] input, double predictionGoal, double alpha, double iterations)
        {
            var result = weights.ToArray();
            for(int iteration = 0; iteration < iterations; ++iteration)
            {
                var prediction = WeightsSum(input, result);
                var error = Math.Pow(prediction - predictionGoal, 2);
                var delta = prediction - predictionGoal;
                var deltaWeight = delta * prediction;
                result = WeightsSub(result, alpha * deltaWeight);
            }
            return result;
        }

        [Test]
        public void ShouldGradientLearnByMultipleInputOneOutput()
        {
            var weights = new double[] { 0.1, 0.2, -0.1 };
            var alpha = 0.01;
            var iterations = 1;

            var toes = new double[] { 8.5, 9.5, 9.9, 9.0 };
            var wlrec = new double[] { 0.65, 0.8, 0.8, 0.9 };
            var nfans = new double[] { 1.2, 1.3, 0.5, 1.0 };

            var winOrLoseData = new double[] {1, 1, 0, -1};

            var predictionGoal = winOrLoseData[0];
            var input = new double[] { toes[0], wlrec[0], nfans[0] };

            var result = GetLearnByGradientMul(weights, input, predictionGoal, alpha, iterations);

            result.SequenceEqual(new double[] { 0.101204, 0.20120400000000002, -0.098796000000000009 }).Should().Be(true);

            //var learnWeights = 
        }
    }
}
