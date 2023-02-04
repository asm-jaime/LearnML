using Keras.Datasets;
using Numpy;

namespace MLTests
{
    public class NDLDigitalRecognizer
    {
        private NDarray _weights01;
        private NDarray _weights12;
        private string _dataPath;

        private static NDarray Relu(NDarray layer) => layer * (layer > 0);
        private static NDarray ReluToDerivative(NDarray layer) => np.where(layer > 0, np.array(1), np.array(0));

        public NDLDigitalRecognizer(string dataPath)
        {
            _weights01 = np.array(0f);
            _weights12 = np.array(0f);
            _dataPath = dataPath;
        }

        public void TrainOnMNISTData()
        {
            var ((x_train, y_train), (x_test, y_test)) = MNIST.LoadData(_dataPath);

            var (images, labels) = (x_train["0:1000"].reshape(1000, 28 * 28) / 255, y_train["0:1000"]);
            var one_hot_labels = np.zeros(labels.len, 10);
            for (var id = 0; id < labels.len; id++)
            {
                var label = labels[id];
                one_hot_labels[id][label] = np.array(1);
            }
            labels = one_hot_labels;

            np.random.seed(1);

            var (alpha, iterations, hidden_size, pixels_per_image, num_labels) = (0.005f, 30, 40, 784, 10);

            _weights01 = 0.2f * np.random.rand(pixels_per_image, hidden_size) - 0.1f;
            _weights12 = 0.2f * np.random.rand(hidden_size, num_labels) - 0.1f;

            for (var i = 0; i < iterations; i++)
            {
                for (var imageId = 0; imageId < images.len; ++imageId)
                {
                    var layer_0 = images[$"{imageId}:{imageId + 1}"];
                    var layer_1 = Relu(np.dot(layer_0, _weights01));
                    var layer_2 = np.dot(layer_1, _weights12);

                    var layer_2_delta = labels[$"{imageId}:{imageId + 1}"] - layer_2;
                    var layer_1_delta = layer_2_delta.dot(_weights12.T) * ReluToDerivative(layer_1);

                    _weights12 += alpha * layer_1.T.dot(layer_2_delta);
                    _weights01 += alpha * layer_0.T.dot(layer_1_delta);
                }
            }
        }

        public int PredictNumberFromImageTest(int testId)
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
            var layer_1 = Relu(np.dot(layer_0, _weights01));
            var layer_2 = np.dot(layer_1, _weights12);
            var result = layer_2.GetData<double>();
            var max = result.Max();
            return result.Select((e, index) => new {e, index}).OrderBy(e => e.e).Last().index;
        }
    }
}
