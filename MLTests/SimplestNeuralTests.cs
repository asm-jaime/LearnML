using FluentAssertions;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLTests
{
    public class SimplestNeuralTests
    {
        [Test]
        public void ShouldSimplePredict()
        {
            // reverse list of number
 
            // {0, 1} принадлежит 0.0000001134
            //[8.5, 9.5, 10, 9]
            // 10 * 0.5 = 5
            SimpiestNeural neuron = new SimpiestNeural() { Weight = 0.1};
            neuron.Prediction(8.5).Should().BeInRange(0.84, 0.86);
            //gameData.GetGameState()
        }

    }
}
