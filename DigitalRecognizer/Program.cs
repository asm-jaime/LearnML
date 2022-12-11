using System;
using System.IO;

namespace DigitalRecognizer
{
    public class Program
    {
        static void Main(string[] args)
        {
            var recognizer = new NDLDigitalRecognizer(Path.GetFullPath(@"D:/projects.active/LearnML/data/mnist.npz"));
            recognizer.TrainOnMNISTData();
            var result = recognizer.PredictNumberFromImageTest(0);
            Console.WriteLine(result);
            Console.ReadLine();
        }
    }
}
