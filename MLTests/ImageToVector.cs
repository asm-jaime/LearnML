using Keras.PreProcessing.Image;
using Numpy;

namespace MLTests
{
    public static class ImageToVector
    {
        public static double[] GetVectorOnPath(string path)
        {
            var image = ImageUtil.LoadImg(path, "grayscale", target_size: (28, 28));
            NDarray x = ImageUtil.ImageToArray(image);
            //x = x.reshape(1);
            //x = x.reshape(78, 78, 3);
            //var shape  = x.shape;
            //x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2]);
            double[] data = x.GetData<double>();
            /*
            var array1 = ImageUtil.ImageToArray(image);

            var array2 = np.expand_dims(array1, axis: 0);
            var array3 = (NDarray)array2;
            double[] data = array3.GetData<double>();
            */
            return data;
            //return new double[] { };
        }
    }
}
