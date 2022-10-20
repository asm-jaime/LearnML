using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Numpy;

namespace MLTests
{
    public class MultipleWeightsNeural
    {
        public NDarray Weights { get; set; }

        public NDarray Prediction(NDarray inputs)
        {
            return inputs.dot(Weights);
        }
    }
}
