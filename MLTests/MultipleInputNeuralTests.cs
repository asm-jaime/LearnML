using FluentAssertions;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Numpy;

namespace MLTests
{
    public class MultipleInputNeuralTests
    {
        [Test]
        public void ShouldMultiplePredict()
        {
            double[] PlayedTime = new double[] { 8.5, 9.5, 9.9, 9.0 };
            double[] PlayedWin = new double[] { 0.65, 0.8, 0.8, 0.9 };
            double[] PlayedFanats = new double[] { 1.2, 1.3, 0.5, 1.0 };

            MultipleWeightsNeural neural = new MultipleWeightsNeural() { Weights = np.array(new double[] {0.1, 0.2, 0}) };

            double result = (double) neural.Prediction(np.array(new double[] { PlayedTime[0], PlayedWin[0], PlayedFanats[0] }));
            result.Should().BeInRange(0.97, 0.99);
        }
    }
}
