using System;

namespace MLTests
{
    internal class SimpiestNeural
    {
        public double Weight { set; get; } = 0;

        public double Prediction(double input) {
            return Weight * input;
        }
    }
}
