using NUnit.Framework;

namespace MLTests
{
    public class MulInMulOutTests
    {
        [Test]
        public void ShouldMulPredict()
        {
            var mmInOut = new MulInMulOut(new double[][] {
                new double[] {0.1, 0.1, -0.3},
                new double[] {0.1, 0.2, 0.0},
                new double[] {0.0, 1.3, 0.1},
            });
        }

    }
}
