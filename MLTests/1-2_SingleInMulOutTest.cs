using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class SingleInMulOut
    {
        double[] _weights;
        public SingleInMulOut(double[] weights)
        {
            _weights = weights;
        }
        public double[] GetPredict(double input)
        {
            return VectorMath.ElementMul(input, _weights);
        }
    }

    public class SingleInMulOutTest
    {
        [Test]
        public void ShouldReturnInjureWinSad()
        {
            double[] PlayedTime = new double[] { 8.5, 9.5, 9.9, 9.0 };
            double[] PlayedWin = new double[] { 0.65, 0.8, 0.8, 0.9 };
            double[] PlayedFanats = new double[] { 1.2, 1.3, 0.5, 1.0 };

            var multiple = new SingleInMulOut(new double[] { 0.3, 0.2, 0.9 });
            var result = multiple.GetPredict(PlayedWin[0]);
            result.SequenceEqual(new double[] { 0.195, 0.13, 0.58500000000000008 }).Should().Be(true);
        }
    }
}
