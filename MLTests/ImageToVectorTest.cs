using FluentAssertions;
using NUnit.Framework;

namespace MLTests
{
    public class ImageToVectorTest
    {
        [Test]
        public void LoadImageTest()
        {
            var projectPath = "D:/projects.active/LearnML";
            var path = Path.GetFullPath(@$"{projectPath}/data/0/1.jpg");
            IImageToVector imgToVector = new ImageToVector();
            var vector = imgToVector.GetVectorOnPath(path);
            vector.Sum().Should().NotBe(0.0f);
        }
    }
}
