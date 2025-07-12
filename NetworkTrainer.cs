namespace NeuralNetwork;

public static class NetworkTrainer
{
    public static SGDOptimizer CreateSGDOptimizer(Network network, Optimizer.LossFunction lossFunction, double baseLearningRate)
    {
        return new SGDOptimizer(network, lossFunction, baseLearningRate);
    }

    public static void Optimize(SGDOptimizer optimizer, double[][] inputs, double[][] outputs, uint epochs)
    {
        optimizer.Optimize(inputs, outputs, epochs);
    }
}
