namespace MLTests
{
    public class SingleInMulOut
    {
        double[] _weights;
        public SingleInMulOut(double[] weights)
        {
            _weights = weights;
        }
        public double[] GetPredict(double input)
        {
            return VectorMath.ElementMul(input, _weights);
        }
    }
}
