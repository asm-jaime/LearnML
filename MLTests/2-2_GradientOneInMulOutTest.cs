using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class GradientOneInMulOutTests
    {
        private double[] ElementMul(double[] vecA, double element)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] * element;
            }
            return result;
        }

        private double[] Mul(double[] vecA, double[] vecB)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] * vecB[i];
            }
            return result;
        }

        private double[] Sub(double[] vecA, double[] vecB)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] - vecB[i];
            }
            return result;

        }

        private double[] SubA(double[] vecA, double[] vecB, double alpha)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] - vecB[i]*alpha;
            }
            return result;

        }

        private double[] GradientLearnByOneInputToMulOut(double[] weights, double input, double[] predictionGoal, double alpha, int iterations)
        {
            var result = weights.ToArray();
            for(int i = 0; i < iterations; i++)
            {
                var predictions = ElementMul(result, input);
                var deltas = Sub(predictions, predictionGoal);
                var deltasWeights = ElementMul(deltas, input);
                result = SubA(result, deltasWeights, alpha);
            }
            return result;
        }

        [Test]
        public void ShouldGradientLearnMulOut()
        {
            var weights = new double[] { 0.3, 0.2, 0.9 };

            var wlrec = new double[] { 0.65, 1.0, 1.0, 0.9 };

            var hurt  = new double[] {0.1, 0.0, 0.0, 0.1};
            var win = new double[] { 1, 1, 0, 1 };
            var sad = new double[] { 0.1, 0.0, 0.1, 0.2 };

            var alpha = 0.2;
            var iterations = 3000;

            var input = wlrec[0];
            var predictionGoal = new double[] { hurt[0], win[0], sad[0] };
            var weightsLearned = GradientLearnByOneInputToMulOut(weights, input, predictionGoal, alpha, iterations);
            var result = ElementMul(weightsLearned, input);
            Assert.That(predictionGoal[0], Is.EqualTo(Math.Round(result[0], 2)));
            Assert.That(predictionGoal[1], Is.EqualTo(Math.Round(result[1], 2)));
            Assert.That(predictionGoal[2], Is.EqualTo(Math.Round(result[2], 2)));
        }
    }
}
