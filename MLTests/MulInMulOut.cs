namespace MLTests
{
    public class MulInMulOut
    {
        double[][] _weights;
        public MulInMulOut(double[][] weights)
        {
            _weights = weights;
        }
        public double[] GetPrediction(double[] vec)
        {
            return VectorMath.VectMatMul(vec, _weights);
        }
    }
}
