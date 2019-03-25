using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNFromScratch
{
    class Neuron
    {
        private Vector<double> weights;
        private double bias;

        public Vector<double> Weights { get => weights; set => weights = value; }
        public double Bias { get => bias; set => bias = value; }

        public Neuron(Vector<double> Weight, double Bias)
        {
            this.Weights = Weight;
            this.Bias = Bias;
        }

        public double Feedback(Vector<double> Input)
        {
            double total = Weights.DotProduct(Input) + Bias;
            double res = Sigmoid(total);
            return res;
        }

        private static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
    }
}
