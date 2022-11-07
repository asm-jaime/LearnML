using FluentAssertions;
using Numpy;
using NUnit.Framework;

namespace MLTests
{
    public class MatrixTests
    {
        [Test]
        public void ShouldMatrixBe()
        {
            var vectorA = np.array(new double[] { 0, 1, 2, 3 });

            var twoOnTwoMatrix = np.zeros(2, 2);
            var twoOnTwoData = twoOnTwoMatrix.GetData<double>();
            twoOnTwoData.Length.Should().Be(4);
            twoOnTwoData.Sum().Should().Be(0);

            var randomMatrix = np.random.rand(2, 5);
            var randomData = randomMatrix.GetData<double>();
            randomData.Length.Should().Be(10);
            randomData.Sum().Should().NotBe(0);

            var weightsA = np.array(new double[,] {
                {1, 2},
                {3, 4},
            });
            var dataWeights = weightsA.GetData<double>();
            dataWeights.SequenceEqual(new double[] { 1, 2, 3, 4 }).Should().Be(true);
        }

        [Test]
        public void ShouldMatrixMultiple()
        {
            var vectorA = np.array(new double[] { 0, 1, 2, 3 });
            var vectorB = np.array(new double[] { 4, 5, 6, 7 });
            var vectorC = np.array(new double[,] { {0, 1, 2, 3 }, {4, 5, 6, 7} });

            var mulVectorA = vectorA * 0.1;
            var mulADouble = mulVectorA.GetData<double>().Select(e => Math.Round(e, 2)).ToArray();
            mulADouble.SequenceEqual(new double[] { 0, 0.1, 0.2, 0.3 }).Should().BeTrue();

            var mulVectorB = vectorA * vectorB;
            mulVectorB.GetData<double>().SequenceEqual(new double[] { 0, 5, 12, 21 }).Should().BeTrue();

            var mulAC = vectorA * vectorC;
            var mulACDouble = mulAC.GetData<double>().Select(e => Math.Round(e, 2)).ToArray();
            mulACDouble.SequenceEqual(new double[] {0, 1, 4, 9, 0, 5, 12, 21}).Should().BeTrue();
        }
    }
}
