using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace NNFromScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] w = new double[2] { 0.0, 1.0 };
            double b = 4.0;
            Vector<double> weights = Vector<double>.Build.Dense(w);
            Neuron neuron = new Neuron(weights, b);
            Vector<double> x = Vector<double>.Build.Dense(new double[2] { 2.0, 3.0 });
            Console.Write("The feedback of the neuron is: "+
                neuron.Feedback(x).ToString() + "\n");

            ExampleNeuronNetwork exampleNeuronNetwork = new ExampleNeuronNetwork();
            Console.Write("The feedback of the neuron network is: " +
                exampleNeuronNetwork.Feedback(x).ToString() + "\n");
            
        }
    }
}
