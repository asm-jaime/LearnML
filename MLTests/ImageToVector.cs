using Keras.PreProcessing.Image;
using Numpy;

namespace MLTests
{
    public interface IImageToVector
    {
        float[] GetVectorOnPath(string path);
    }

    public class ImageToVector : IImageToVector
    {
        private const string DefaultImageType = "grayscale";
        private static readonly (int, int) DefaultSize = (28, 28);

        public ImageToVector()
        {

        }
        float[] GetNormalizedVector(float[] vector)
        {
            var max = vector.Max();
            var min = vector.Min();
            for(var i = 0; i < vector.Length; i++) vector[i] /= max;
            return vector;
        }

        public float[] GetVectorOnPath(string path)
        {
            var image = ImageUtil.LoadImg(path, DefaultImageType, target_size: DefaultSize);
            NDarray imageArray = ImageUtil.ImageToArray(image);
            return GetNormalizedVector(imageArray.GetData<float>());
        }
    }
}
