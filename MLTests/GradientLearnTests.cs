using FluentAssertions;
using Keras.Optimizers;
using NUnit.Framework;
using System.Diagnostics;

namespace MLTests
{
    public class GradientLearnTests
    {
        private double LearnWeightByGradient(double weight, double input, double predictionGoal, int iterations)
        {
            var result = weight;
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                var prediction = input * result;
                var delta = prediction - predictionGoal;
                var weightDelta = delta * input;
                result = result - weightDelta;
            }
            return result;
        }

        [Test]
        public void ShouldLearnByGradient()
        {
            var weight = 0.5;
            var input = 0.5;
            var predictionGoal = 0.8;
            var iterations = 20;

            var learnedWeight = LearnWeightByGradient(weight, input, predictionGoal, iterations);
            var result = Math.Round(learnedWeight * input, 2);
            result.Should().Be(0.8);
        }

        [Test]
        public void ShouldLearnByGradientOneIterationAndAlphaKoef()
        {
            var weight = 0.1;
            var alpha = 0.01;
            var input = 8.5;
            var predictionGoal = 1;
            var prediction = weight * input;
            var delta = prediction - predictionGoal;
            var weightDelta = delta * input;

            var learnedWeight = weight - weightDelta * alpha;

            var result = Math.Round(learnedWeight * input, 1);
            result.Should().Be(1);
        }

        [Test]
        public void ShouldNotLearnWeightJustIn4Iteration()
        {
            var (weight, input, predictionGoal, iterations) = (0.0, 0.5, 0.8, 4);
            var learnedWeight = LearnWeightByGradient(weight,input, predictionGoal, iterations);
            var result = Math.Round(learnedWeight * input, 2);
            result.Should().NotBe(0.8);
        }

        [Test]
        public void ShouldLearnWeightIn4Iteration2()
        {
            var (weight, input, predictionGoal, iterations) = (0.0, 1.1, 0.8, 4);
            var learnedWeight = LearnWeightByGradient(weight,input, predictionGoal, iterations);
            var result = Math.Round(learnedWeight * input, 2);
            result.Should().Be(0.8);
        }
    }
}
