namespace NeuralNetwork.Optimizers;

public class AdamOptimizer : IOptimizer
{
    private Network _network;
    private LossType _lossType;
    private double _learningRate;
    
    public AdamOptimizer(Network network, LossType lossFunction, double baseLearningRate)
    {
        _network = network;
        _lossType = lossFunction;
        _learningRate = baseLearningRate;
    }

    public void Optimize(double[][] inputs, double[][] outputs, uint totalEpochs)
    {
        throw new NotImplementedException();
    }

    public double[] OptimizeTracked(double[][] inputs, double[][] outputs, uint totalEpochs)
    {
        throw new NotImplementedException();
    }
    
    //TODO: Actually implement this shit.
}