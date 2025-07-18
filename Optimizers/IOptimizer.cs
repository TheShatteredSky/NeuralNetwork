namespace NeuralNetwork.Optimizers;

internal interface IOptimizer
{
    void Optimize(double[][] inputs, double[][] outputs, uint totalEpochs);
    double[] OptimizeTracked(double[][] inputs, double[][] outputs, uint totalEpochs);
}