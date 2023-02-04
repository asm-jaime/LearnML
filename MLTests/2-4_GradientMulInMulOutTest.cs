using NUnit.Framework;

namespace MLTests
{
    public class GradientMulInMulOutTest
    {
        private double[] MulVV(double[] vecA, double[] vecB)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] * vecB[i];
            }
            return result;
        }
        private double MulScalarVV(double[] vecA, double[] vecB)
        {
            double result = 0;
            for (int i = 0; i < vecA.Length; i++)
            {
                result = result + vecA[i] * vecB[i];
            }
            return result;
        }

        public double[] MulScalarVM(double[] vecA, double[][] matrixA)
        {
            var result = new double[vecA.Length];
            for (var index = 0; index < result.Length; ++index)
            {
                result[index] = MulScalarVV(vecA, matrixA[index]);
            }
            return result;
        }

        private double[] SubVV(double[] vecA, double[] vecB)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] - vecB[i];
            }
            return result;

        }

        private double[] SubVE(double[] vecA, double element)
        {
            var result = new double[vecA.Length];
            for (int i = 0; i < vecA.Length; i++)
            {
                result[i] = vecA[i] - element;
            }
            return result;

        }

        private double[] GradientLearnMulInOut(double[][] weights, double[] input, double[] predictionGoal, double alpha, int iterations)
        {
            var result = weights.Select(a => a.ToArray()).ToArray();
            for(var i = 0; i < iterations; ++i)
            {
                var prediction = MulScalarVM(input, result);
                var deltas = SubVV(prediction, predictionGoal);
                var derivatives = MulVV(deltas, input);
                result[0] = SubVE(result[0], derivatives[0] * alpha);
                result[1] = SubVE(result[1], derivatives[1] * alpha);
                result[2] = SubVE(result[2], derivatives[2] * alpha);
            }

            return MulScalarVM(input, result);
        }

        [Test]
        public void ShouldGradientLearnFromMulInToMulOut()
        {
            var weights = new double[][] {
                new double[] {0.1, 0.1, -0.3},
                new double[] {0.1, 0.2, 0.0},
                new double[] {0.0, 1.3, 0.1},
            };
            var alpha = 0.01;
            var iterations = 100;

            var toes = new double[] { 8.5, 9.5, 9.9, 9.0 };
            var wlrec = new double[] { 0.65, 0.8, 0.8, 0.9 };
            var nfans = new double[] { 1.2, 1.3, 0.5, 1.0 };

            var hurt = new double[] { 0.1, 0.0, 0.0, 0.1 };
            var win = new double[] { 1.0, 1.0, 0.0, 1.0 };
            var sad = new double[] { 0.1, 0.0, 0.1, 0.2 };

            var input = new double[] { toes[0], wlrec[0], nfans[0] };
            var predictionGoal = new double[] { hurt[0], win[0], sad[0] };

            var result = GradientLearnMulInOut(weights, input, predictionGoal, alpha, iterations);

            Assert.That(Math.Round(result[0], 2), Is.EqualTo(predictionGoal[0]));
            Assert.That(Math.Round(result[1], 2), Is.EqualTo(predictionGoal[1]));
            Assert.That(Math.Round(result[2], 2), Is.EqualTo(predictionGoal[2]));
        }
    }
}
