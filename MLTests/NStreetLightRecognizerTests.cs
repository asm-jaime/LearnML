using FluentAssertions;
using Numpy;
using NUnit.Framework;

namespace MLTests
{
    public class NStreetLightRecognizerTests
    {
        private NDarray NGradient(NDarray weights, NDarray input, NDarray predictionGoal, float alpha, int iterations)
        {
            var resultWeights = weights.copy();
            for(int i = 0; i < iterations; i++)
            {
                var prediction = input.dot(resultWeights);
                var delta = prediction - predictionGoal ;
                resultWeights = resultWeights - alpha * (input * delta);
            }
            return resultWeights;
        }

        [Test]
        public void ShouldPredictWalkOrStayOnDifferentLight()
        {
            var stop = 0f;
            var walk = 1f;

            var streetLights = np.array(new float[,] {
                {1f, 0f, 1f},
                {0f, 1f, 1f},
                {0f, 0f, 1f},
                {1f, 1f, 1f},
                {0f, 1f, 1f},
                {1f, 0f, 1f},
            });
            var stayOrWalk = np.array(new float[] { 0f, 1f, 0f, 1f, 1f, 0f });
            var weights = np.array(new float[] { 0.5f, 0.48f, -0.7f });
            var alpha = 0.1f;

            var input = streetLights[0];
            var predictionGoal = stayOrWalk[0];

            var iterations = 100;

            var weightsResult = NGradient(weights, input, predictionGoal, alpha, iterations);
            var result = (float) input.dot(weightsResult);
            result.Should().BeInRange(-0.0001f, 0.0001f);
        }
    }
}
