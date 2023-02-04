using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class SimpiestNeural
    {
        public double Weight { set; get; } = 0;

        public double Prediction(double input) {
            return Weight * input;
        }
    }

    public class SimplestNeuralTests
    {
        [Test]
        public void ShouldSimplePredict()
        {
            // reverse list of number

            // {0, 1} принадлежит 0.0000001134
            //[8.5, 9.5, 10, 9]
            // 10 * 0.5 = 5
            SimpiestNeural neuron = new SimpiestNeural() { Weight = 0.1 };
            neuron.Prediction(8.5).Should().BeInRange(0.84, 0.86);
        }

    }
}
