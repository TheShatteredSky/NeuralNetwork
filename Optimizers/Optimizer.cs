using NeuralNetwork.Core;

namespace NeuralNetwork.Optimizers;

public class Optimizer
{
    internal LossFunction LossFunct;
    internal double BaseLearningRate;
    internal Network Network;
    
    public enum LossFunction
    {
        CrossEntropy, 
        MSE
    }

    internal Optimizer(Network network, LossFunction lossFunction, double baseLearningRate)
    {
        Network = network;
        LossFunct = lossFunction;
        BaseLearningRate = baseLearningRate;
    }

    public virtual void Optimize(double[][] inputs, double[][] outputs, uint epochs)
    {
        throw new NotImplementedException();
    }
}