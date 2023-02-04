using FluentAssertions;
using Numpy;
using NUnit.Framework;

namespace MLTests
{
    public class MulInMulOut
    {
        double[][] _weights;
        public MulInMulOut(double[][] weights)
        {
            _weights = weights;
        }
        public double[] GetPrediction(double[] vec)
        {
            return VectorMath.VectMatMul(vec, _weights);
        }
    }

    public class MulInMulOutTest
    {
        [Test]
        public void ShouldMulPredict()
        {
            var mmInOut = new MulInMulOut(new double[][] {
                new double[] {0.1, 0.1, -0.3},
                new double[] {0.1, 0.2, 0.0},
                new double[] {0.0, 1.3, 0.1},
            });
            var result = mmInOut.GetPrediction(new double[] { 8.5, 0.65, 1.2 });
            result.SequenceEqual(new double[] { 0.555, 0.98000000000000009, 0.96500000000000008 }).Should().Be(true);
        }

        [Test]
        public void ShouldImposeNetworks()
        {
            var firstNetwork = new MulInMulOut(new double[][] {
                new double[] {0.1, 0.2, -0.1},
                new double[] {-0.1, 0.1, 0.9},
                new double[] {0.1, 0.4, 0.1},
            });
            var secondNetwork = new MulInMulOut(new double[][] {
                new double[] {0.3, 1.1, -0.3},
                new double[] {0.1, 0.2, 0.0},
                new double[] {0.0, 1.3, 0.1},
            });
            var input = new double[] { 8.5, 0.65, 1.2 };

            var intermediateResult = firstNetwork.GetPrediction(input);
            var result = secondNetwork.GetPrediction(intermediateResult);

            result.SequenceEqual(new double[] { 0.21350000000000002, 0.14500000000000002, 0.5065 }).Should().Be(true);
        }

        [Test]
        public void ShouldImposeNetworksByNumpy()
        {
            /*
            var weightsA = np.array(new double[,] {
                {0.1, -0.1, 0.1},
                {0.2, 0.1, 0.4},
                {-0.1, 0.9, 0.1},
            });
            var weightsB = np.array(new double[,] {
                {0.3, 0.1, 0.0},
                {1.1, 0.2, 1.3},
                {-0.3, 0.0, 0.1},
            });
            */

            var weightsA = np.array(new double[,] {
                {0.1, 0.2, -0.1},
                {-0.1, 0.1, 0.9},
                {0.1, 0.4, 0.1},
            }).T;
            var weightsB = np.array(new double[,] {
                {0.3, 1.1, -0.3},
                {0.1, 0.2, 0.0},
                {0.0, 1.3, 0.1},
            }).T;

            var input = np.array(new double[] { 8.5, 0.65, 1.2 });

            var hid = input.dot(weightsA);
            var result = hid.dot(weightsB);
            result.GetData<double>().SequenceEqual(new double[] { 0.21350000000000002, 0.14500000000000002, 0.5065 }).Should().Be(true);

            //(double[])result.SequenceEqual(new double[] { 0.21350000000000002, 0.14500000000000002, 0.5065 }).Should().Be(true);
        }
    }
}
