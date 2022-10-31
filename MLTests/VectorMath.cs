﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLTests
{
    public static class VectorMath
    {

        public static double[] ElementwiseMultiplication(double[] vecA, double[] vecB)
        {
            return vecA.Select((elementA, index) => elementA * vecB[index]).ToArray();
        }

        public static double[] ElementwiseAddition(double[] vecA, double[] vecB)
        {
            return vecA.Select((elementA, index) => elementA + vecB[index]).ToArray();
        }

        public static double ElementwiseSum(double[] vecA)
        {
            return vecA.Sum();
        }

        public static double ElementwiseAverage(double[] vecA)
        {
            return vecA.Sum() / vecA.Length;
        }
    }
}