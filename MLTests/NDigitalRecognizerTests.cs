using NUnit.Framework;

namespace MLTests
{
    public class NDigitalRecognizerTests
    {
        [Test]
        public void ShouldRecognizeDigit()
        {
            var projectPath = "D:/projects.active/LearnML";
            var trainDataPath = Path.GetFullPath(@$"{projectPath}/data");
            var digitZeroPath = Path.Combine(trainDataPath, "0", "155.jpg");
            var digitOnePath = Path.Combine(trainDataPath, "1", "1055.jpg");
            var digitTwoPath = Path.Combine(trainDataPath, "2", "1055.jpg");
            var digitThreePath = Path.Combine(trainDataPath, "3", "1055.jpg");
            var digitFourPath = Path.Combine(trainDataPath, "4", "1055.jpg");
            var digitFivePath = Path.Combine(trainDataPath, "5", "1055.jpg");
            var digitSixPath = Path.Combine(trainDataPath, "6", "1055.jpg");
            var digitSevenPath = Path.Combine(trainDataPath, "7", "1055.jpg");
            var digitEightPath = Path.Combine(trainDataPath, "8", "1055.jpg");
            var digitNinePath = Path.Combine(trainDataPath, "9", "1055.jpg");

            var inputSize = 28 * 28;
            var outputSize = 10;

            var alpha = 0.001f;
            var iterations = 20;
            var sampleSize = 100;

            var vectorReader = new ImageToVector();
            var recognizer = new NDigitalRecognizer(vectorReader, inputSize, outputSize, trainDataPath, sampleSize, alpha, iterations);

            for (int learnDigit = 0; learnDigit < 10; learnDigit++)
            {
                recognizer.Learn(learnDigit);
            }
            for (int learnDigit = 0; learnDigit < 10; learnDigit++)
            {
                recognizer.Learn(learnDigit);
            }
            for (int learnDigit = 0; learnDigit < 10; learnDigit++)
            {
                recognizer.Learn(learnDigit);
            }

            {
                var (digit, value) = recognizer.Predict(digitNinePath);
                Assert.That(digit, Is.EqualTo(9));
                Assert.That(value, Is.AtLeast(0.5f));
            }
            {
                var (digit, value) = recognizer.Predict(digitFivePath);
                Assert.That(digit, Is.EqualTo(5));
                Assert.That(value, Is.AtLeast(0.5f));

            }

            //recognizer.Predict(digitOnePath)[zero].Should().BeInRange(0.0f, 0.6f);
            //recognizer.Predict(digitZeroPath)[zero].Should().BeInRange(0.8f, 1.2f);
        }
    }
}
