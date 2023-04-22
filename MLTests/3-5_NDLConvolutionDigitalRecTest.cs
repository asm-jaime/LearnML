using FluentAssertions;
using Keras.Datasets;
using Numpy;
using NUnit.Framework;

namespace MLTests
{
    public class NDLConvolutionDigitalRec
    {
        private NDarray _kernels;
        private NDarray _weights12;
        private string _dataPath;

        private static NDarray Tanh(NDarray layer) => np.tanh(layer);
        private static NDarray TanhToDerivative(NDarray layer) => 1 - np.power(layer, np.array(2));
        private static NDarray Softmax(NDarray layer)
        {
            var temp = np.exp(layer);
            return temp / np.sum(temp, axis: 1, keepdims: true);
        }

        public NDarray GetImageSection(NDarray layer, int rowFrom, int rowTo, int colFrom, int colTo)
        {
            var subsection = layer[$":,{rowFrom}:{rowTo},{colFrom}{colTo}"];
            return subsection.reshape(-1, 1, rowTo - rowFrom, colTo - colTo);
        }

        public NDLConvolutionDigitalRec(string dataPath)
        {
            _kernels = np.array(0f);
            _weights12 = np.array(0f);
            _dataPath = dataPath;
        }

        public void TrainOnMNISTDataWhithTath()
        {
            var ((x_train, y_train), (_, _)) = MNIST.LoadData(_dataPath);

            var (images, labels) = (x_train["0:1000"].reshape(1000, 28 * 28) / 255, y_train["0:1000"]);
            var one_hot_labels = np.zeros(labels.len, 10);
            for (var id = 0; id < labels.len; id++)
            {
                var label = labels[id];
                one_hot_labels[id][label] = np.array(1);
            }
            labels = one_hot_labels;

            np.random.seed(1);

            var (alpha, iterations) = (2.0f, 30);
            var (pixels_per_image, num_labels) = (784, 10);
            var batch_size = 128;
            var input_rows = 28;
            var input_cols = 28;

            var kernel_rows = 3;
            var kernel_cols = 3;
            var num_kernels = 16;

            var hidden_size = (input_rows - kernel_rows) * (input_cols - kernel_cols) * num_kernels;

            _kernels = 0.02f * np.random.rand(kernel_rows * kernel_cols, num_kernels) - 0.01f;
            _weights12 = 0.2f * np.random.rand(hidden_size, num_labels) - 0.1f;

            for (var i = 0; i < iterations; i++)
            {
                var correct_cnt = 0;
                for (var imageId = 0; imageId < images.len / batch_size; ++imageId)
                {
                    var (batch_start, batch_end) = (imageId * batch_size, (imageId + 1) * batch_size);
                    var layer_0 = images[$"{batch_start}:{batch_end}"];
                    layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28);
                    var shape = layer_0.shape;

                    var layer_1 = Tanh(np.dot(layer_0, _kernels));
                    var dropout_mask = np.random.randint(2, null, layer_1.shape.Dimensions);
                    layer_1 *= dropout_mask * 2;
                    var layer_2 = Softmax(np.dot(layer_1, _weights12));

                    var layer_2_delta = (labels[$"{batch_start}:{batch_end}"] - layer_2) / (batch_size * layer_2.shape[0]);
                    var layer_1_delta = layer_2_delta.dot(_weights12.T) * TanhToDerivative(layer_1);
                    layer_1_delta *= dropout_mask;


                    _weights12 += alpha * layer_1.T.dot(layer_2_delta);
                    _kernels += alpha * layer_0.T.dot(layer_1_delta);
                }
            }
        }

        public int PredictNumber(int testId)
        {
            var ((x_train, y_train), (x_test, y_test)) = MNIST.LoadData(_dataPath);

            var test_images = x_test.reshape(x_test.len, 28 * 28) / 255;
            var test_labels = np.zeros(y_test.len, 10);
            for (var id = 0; id < y_test.len; id++)
            {
                var test = y_test[id];
                test_labels[id][test] = np.array(1);
            }

            var layer_0 = test_images[$"{testId}:{testId + 1}"];
            var layer_1 = Tanh(np.dot(layer_0, _kernels));
            var layer_2 = np.dot(layer_1, _weights12);
            var result = layer_2.GetData<double>();
            var max = result.Max();
            return result.Select((e, index) => new { e, index }).OrderBy(e => e.e).Last().index;
        }

    }

    public class NDLConvolutionDigitalRecTest
    {
        [Test]
        public void ShouldPredictByUsingTathActivationFunc()
        {
            var recognizer = new NDLConvolutionDigitalRec(Path.GetFullPath(@"D:/projects.active/LearnML/data/mnist.npz"));
            recognizer.TrainOnMNISTDataWhithTath();
            var result = recognizer.PredictNumber(0);
            result.Should().Be(7);
        }
    }
}
