using MLTests;

//var vector = ImageToVector.GetVectorOnPath(@"./data/0/1.jpg");
var recognizer = new NDLDigitalRecognizer(Path.GetFullPath(@"D:/projects.active/LearnML/data/mnist.npz"));
recognizer.TrainOnMNISTData();
var result = recognizer.PredictNumberFromImageTest(0);
Console.WriteLine(result);
Console.ReadLine();
