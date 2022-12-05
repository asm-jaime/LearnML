using FluentAssertions;
using Numpy;
using NUnit.Framework;

namespace MLTests
{
    public class NSteerLightReluTests
    {
        private NDarray Relu(NDarray layer)
        {
            var shape = layer.shape;
            var data = layer.GetData<double>();
            var result = data.Select(num => num < 0f ? 0f : (float) num).ToArray();
            return np.array(result).reshape(shape);
        }

        private NDarray ReluToDerivative(NDarray layer)
        {
            var shape = layer.shape;
            return np.array(layer.GetData<float>().Select(num => num > 0f ? 1f : 0f).ToArray()).reshape(shape);
        }

        [Test]
        public void ShouldLearnOnNoncorrelationData()
        {

            var streetLights = np.array(new float[,] {
                {1f, 0f, 1f},
                {0f, 1f, 1f},
                {0f, 0f, 1f},
                {1f, 1f, 1f},
            });
            var stayOrWalk = np.array(new float[,] { { 1f, 1f, 0f, 0f } }).T;
            var alpha = 0.2f;
            var inputLayerSize = 3;
            var hiddenLayerSize = 4;
            var outputLayerSize = 1;

            var weights01 = 2 * np.random.rand(inputLayerSize, hiddenLayerSize) - 1;
            var weights12 = 2 * np.random.rand(hiddenLayerSize, outputLayerSize) - 1;

            /*
            var layer0 = streetLights[0];
            var layer1 = Relu(np.dot(layer0, weights01));
            var layer2 = np.dot(layer1, weights12);
            */
            /*
            var prediction = weight*input
            var delta = prediction - predictionGoal;
            result = result + delta * alpha * input;
             */

            var iterations = 60;
            for (var iteration = 0; iteration < iterations; iteration++)
            {
                for (var sampleId = 0; sampleId < streetLights.len; sampleId++)
                {
                    var layer0 = streetLights[$"{sampleId}:{sampleId + 1}"];
                    var layer1 = Relu(np.dot(layer0, weights01));
                    var layer2 = np.dot(layer1, weights12);

                    var layer2Delta = stayOrWalk[$"{sampleId}:{sampleId + 1}"] - layer2;
                    var layer1Delta = layer2Delta.dot(weights12.T) * ReluToDerivative(layer1);

                    weights12 = weights12 + alpha * layer1.T.dot(layer2Delta);
                    weights01 = weights01 + alpha * layer0.T.dot(layer1Delta);

                    //weights12 += alpha * layer1.T.dot();
                }
            }

            var resLayer0 = streetLights[$"{2}:{2 + 1}"];
            var resLayer1 = Relu(np.dot(resLayer0, weights01));
            var resLayer2 = np.dot(resLayer1, weights12);
        }


        [Test]
        public void ShouldLearnByEmbeddedReluData()
        {

            var streetLights = np.array(new float[,] {
                {1f, 0f, 1f},
                {0f, 1f, 1f},
                {0f, 0f, 1f},
                {1f, 1f, 1f},
            });
            var stayOrWalk = np.array(new float[,] { { 1f, 1f, 0f, 0f } }).T;
            var alpha = 0.2f;
            var inputLayerSize = 3;
            var hiddenLayerSize = 4;
            var outputLayerSize = 1;

            var weightsInputHidden = 2*np.random.rand(inputLayerSize, hiddenLayerSize) - 1;
            var weightsHiddenOutput = 2*np.random.rand(hiddenLayerSize, outputLayerSize) - 1;

            var iterations = 60;

            for(int i = 1; i <= iterations; ++i)
            {
                for(int sampleId = 0; sampleId < streetLights.len; ++sampleId)
                {
                    var streetLight = streetLights[$"{sampleId}:{sampleId+1}"];
                    var layerInput = streetLight;
                    var layerHidden = np.dot(layerInput, weightsInputHidden);
                    var layerHiddenRelu = layerHidden * (layerHidden > 0);
                    var layerOutput = np.dot(layerHiddenRelu, weightsHiddenOutput);

                    var deltaOutputHidden = stayOrWalk[$"{sampleId}:{sampleId + 1}"] - layerOutput;
                    weightsHiddenOutput += alpha * layerHiddenRelu.T.dot(deltaOutputHidden);

                    var layerHiddenReluMultiplayer = np.where(layerHiddenRelu > 0, np.array(1), np.array(0));
                    var deltaInputHidden = deltaOutputHidden.dot(weightsHiddenOutput.T) * layerHiddenReluMultiplayer;
                    weightsInputHidden += alpha * layerInput.T.dot(deltaInputHidden);
                }
            }

            var inputLayer = streetLights[$"{2}:{2 + 1}"];
            var hiddenLayer = np.dot(inputLayer, weightsInputHidden);
            var outputLayer = np.dot(hiddenLayer * (hiddenLayer > 0), weightsHiddenOutput);
            float data = outputLayer.GetData<float>().FirstOrDefault();
            data.Should().BeInRange(-0.05f, 0.05f);
        }
    }
}
