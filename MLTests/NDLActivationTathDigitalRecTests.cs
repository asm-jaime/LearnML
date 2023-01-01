using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class NDLActivationTathDigitalRecTests
    {
        [Test]
        public void ShouldPredictByUsingTathActivationFunc()
        {
            var recognizer = new NDLActivationTathDigitalRec(Path.GetFullPath(@"D:/projects.active/LearnML/data/mnist.npz"));
            recognizer.TrainOnMNISTDataWhithTath();
            var result = recognizer.PredictNumber(0);
            result.Should().Be(7);
        }

    }
}
