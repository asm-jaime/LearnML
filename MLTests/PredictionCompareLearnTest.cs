using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class PredictionCompareLearnTest
    {
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
    }
}
