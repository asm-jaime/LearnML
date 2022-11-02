namespace MLTests
{
    public class MultipleInputsMultipleOutputs
    {
        double[] _weights;
        public MultipleInputsMultipleOutputs(double[] weights)
        {
            _weights = weights;
        }
        public double[] GetPredict(double input)
        {
            return VectorMath.ElementMul(input, _weights);
        }
    }
}
