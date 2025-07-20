namespace NeuralNetwork.Optimizers;

internal interface IOptimizer
{
    void Optimize(Dataset data, uint totalEpochs);
    double[] OptimizeTracked(Dataset data, uint totalEpochs);
}