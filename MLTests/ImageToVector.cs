using Keras.PreProcessing.Image;
using Numpy;

namespace MLTests
{
    public static class ImageToVector
    {
        private static float[] GetNormalizedVector(float[] vector)
        {
            var max = vector.Max();
            var min = vector.Min();
            for(var i = 0; i < vector.Length; i++) vector[i] /= max;
            return vector;
        }

        public static float[] GetVectorOnPath(string path)
        {
            var image = ImageUtil.LoadImg(path, "grayscale", target_size: (28, 28));
            NDarray x = ImageUtil.ImageToArray(image);
            var shape = x.shape; // (28, 28, 3)
            //var norm = new Keras.Layers.BatchNormalization();
            
            //x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2]);
            float[] data = x.GetData<float>();
            /*
            var array1 = ImageUtil.ImageToArray(image);

            var array2 = np.expand_dims(array1, axis: 0);
            var array3 = (NDarray)array2;
            float[] data = array3.GetData<float>();
            */
            return GetNormalizedVector(data);
            //return new float[] { };
        }
    }
}
