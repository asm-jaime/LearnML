using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class PredictionCompareLearnTest
    {
        public double GetPredictFromSimpleNetwork(double input, double weight)
        {
            var prediction = input * weight;
            return prediction;
        }

        public double GetLearnNetwork(double weight, double error, double errorUp, double errorDn, double step)
        {
            var result = weight;
            if(error > errorUp || error > errorDn)
            {
                if (errorDn < errorUp) result = result - step;
                if (errorUp < errorDn) result = result + step;
            }
            return result;
        }

        public double LearnWeightInNIteration(double weight, double input, double predictionGoal, double step, int iterations)
        {
            var result = weight;
            for(int iteration = 0; iteration < iterations; iteration++)
            {
                var prediction = input * result;
                var error = Math.Pow(prediction - predictionGoal, 2);

                var predictionUp = input * (result + step);
                var errorUp = Math.Pow(predictionUp - predictionGoal , 2);

                var predictionDn = input * (result - step);
                var errorDn = Math.Pow(predictionDn - predictionGoal , 2);

                if (errorDn < errorUp) result = result - step;
                if (errorUp < errorDn) result = result + step;
            }
            return result;
        }

        [Test]
        public void ShouldGetSimpleLearnError()
        {
            var weight = 0.5;
            var input = 0.5;

            var goal = 0.8;

            var prediction = input * weight;

            var error = Math.Round(Math.Pow(goal - prediction, 2), 2);
            error.Should().Be(0.3);
        }

        [Test]
        public void ShouldLearnByHotColdMethod()
        {
            var weight = 0.1;

            var toes = 8.5;
            var winOrLose = 1;

            var input = toes;
            var goal = winOrLose;

            var prediction = GetPredictFromSimpleNetwork(input, weight);
            var error = Math.Round(Math.Pow(prediction - goal, 2), 3);
            error.Should().Be(0.022);

            var step = 0.01;
            var predictUp = GetPredictFromSimpleNetwork(input, weight + step);
            var errorUp = Math.Round(Math.Pow(predictUp - goal, 2), 3);
            errorUp.Should().Be(0.004);

            step = 0.01;
            var predictDn = GetPredictFromSimpleNetwork(input, weight - step);
            var errorDn = Math.Round(Math.Pow(predictDn - goal, 2), 3);
            errorDn.Should().Be(0.055);

            var betterWeight = GetLearnNetwork(weight, error, errorUp, predictDn, step);
            var betterPrediction = GetPredictFromSimpleNetwork(input, betterWeight);
            var betterError = Math.Round(Math.Pow(betterPrediction - goal, 2), 3);
            betterError.Should().Be(0.004);
        }

        [Test]
        public void ShouldIterativeLearn()
        {
            var weight = 0.5;
            var input = 0.5;
            var predictionGoal = 0.8;
            var step = 0.001;
            var iterations = 1101;

            var goodWeight = LearnWeightInNIteration(weight, input, predictionGoal, step, iterations);
            var result = Math.Round(input * goodWeight, 2);
            result.Should().Be(0.8);
        }


    }
}
