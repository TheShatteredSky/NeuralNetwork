namespace NeuralNetwork.Optimizers;

public class Optimizer
{    
    
    internal readonly Network Network;
    internal readonly LossFunction LossFunct;
    internal double BaseLearningRate;

    public enum OptimizerType
    {
        SGD,
        Adam
    }
    
    public enum LossFunction
    {
        BinaryCrossEntropy, 
        CategoricalCrossEntropy,
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