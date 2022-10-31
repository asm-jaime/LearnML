using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class VectorMathTests
    {
        [Test]
        public void TestElementWiseMultiplication()
        {
            var A = new double[] { 0, 1, 0, 1 };
            var B = new double[] { 1, 0, 1, 0 };
            var C = new double[] { 0, 1, 1, 0 };
            var D = new double[] { 0.5, 0, 0.5, 0 };
            var E = new double[] { 0, 1, -1, 0 };

            VectorMath.ElementwiseSum(VectorMath.ElementwiseMultiplication(C, E)).Should().Be(0.0);
            VectorMath.ElementwiseSum(VectorMath.ElementwiseMultiplication(C, C)).Should().Be(2);
        }
    }
}
