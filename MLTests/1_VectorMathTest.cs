using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public static class VectorMath
    {
        public static double[] ElementwiseMultiplication(double[] vecA, double[] vecB)
        {
            return vecA.Select((elementA, index) => elementA * vecB[index]).ToArray();
        }

        public static double[] VectMatMul(double[] vecA, double[][] matrix)
        {
            return vecA.Select((_, index) => ElementwiseMultiplication(vecA, matrix[index]).Sum()).ToArray();
        }

        public static double[] ElementMul(double element, double[] vec)
        {
            var result = new double[vec.Length];
            return result.Select((_, index) => vec[index] * element).ToArray();
        }

        public static double[] ElementwiseAddition(double[] vecA, double[] vecB)
        {
            return vecA.Select((elementA, index) => elementA + vecB[index]).ToArray();
        }

        public static double ElementwiseSum(double[] vecA)
        {
            return vecA.Sum();
        }

        public static double ElementwiseAverage(double[] vecA)
        {
            return vecA.Sum() / vecA.Length;
        }
    }

    public class VectorMathTest
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
