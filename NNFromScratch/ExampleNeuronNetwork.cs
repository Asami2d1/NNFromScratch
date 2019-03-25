using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNFromScratch
{
    class ExampleNeuronNetwork
    {
        private static Vector<double> weigths;
        private static double bias;
        public static Vector<double> Weights { get => weigths; set => weigths = value; }
        public static double Bias { get => bias; set => bias = value; }

        private readonly Neuron neuH1;
        private readonly Neuron neuH2;
        private readonly Neuron neuO1;

        public ExampleNeuronNetwork()
        {
            Vector<double> weights = Vector<double>.Build.Dense(new double[2] { 0.0, 1.0 });
            double bias = 0.0;
            Weights = weights;
            Bias = bias;
            neuH1 = new Neuron(Weights, Bias);
            neuH2 = new Neuron(Weights, Bias);
            neuO1 = new Neuron(Weights, Bias);
        }

        public double Feedback(Vector<double> x)
        {
            double outH1 = neuH1.Feedback(x);
            double outH2 = neuH2.Feedback(x);
            double outO1 = neuO1.Feedback(Vector<double>.Build.Dense(new double[2] { outH1, outH2 }));

            return outO1;
        }
    }
}
