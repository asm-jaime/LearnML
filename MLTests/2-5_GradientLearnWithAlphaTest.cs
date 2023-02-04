using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class GradientLearnWithAlphaTest
    {
        private double LearnWeightByGradientAlpha(double weight, double input, double predictionGoal, double alpha, int iterations)
        {
            var result = weight;
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                var prediction = input * result;
                var error = Math.Pow(prediction - predictionGoal, 2);
                var delta = prediction - predictionGoal;
                var weightDelta = delta * input;
                result = result - alpha * weightDelta;
            }
            return result;
        }

        private double FunctionLearnWeight(double weight, double input, double predictionGoal, double alpha, int iterations)
        {
            var result = weight;
            for(int iteration = 0; iteration < iterations; iteration++)
            {
                var prediction = result * input;
                var error = Math.Pow(prediction - predictionGoal, 2);
                var derivative = (prediction - predictionGoal) * input;
                result = result - alpha * derivative;
            }
            return result;
        }

        [Test]
        public void ShouldNotBreakLearnWeight()
        {
            var (weight, input, predictionGoal, alpha, iterations) = (0.0, 2.0, 0.8, 0.1, 10);
            var learnedWeight = FunctionLearnWeight(weight, input, predictionGoal, alpha, iterations);
            var result = Math.Round(learnedWeight * input, 2);
            result.Should().Be(0.8);
        }
    }
}
