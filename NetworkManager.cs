
namespace NeuralNetwork;

using System.Collections.Concurrent;

public class NetworkManager
{
    private string _name;

    private ushort _layerCount;
    private ushort _inputSize;
    private Node.ActivationType _inputType;
    private ushort _hiddenSize;
    private Node.ActivationType _hiddenType;
    private ushort _outputSize;
    private Node.ActivationType _outputType;

    private Network.LossFunction _lossFunction;
    private double _learningRate;

    private Network _network;

    public NetworkManager(Network network)
    {
        _name = network.GetName();
        _lossFunction = network.GetLossFunction();
        _learningRate = network.GetLearningRate();
        _layerCount = network.GetLayerCount();
        _inputSize = network.GetLayers()[0].GetSize();
        _hiddenSize = network.GetLayers()[1].GetSize();
        _outputSize = network.GetLayers()[_layerCount - 1].GetSize();
        _inputType = (Node.ActivationType)network.GetLayers()[0].GetNodes()[0].GetActivation()!;
        _hiddenType = (Node.ActivationType)network.GetLayers()[1].GetNodes()[0].GetActivation()!;
        _outputType = (Node.ActivationType)network.GetLayers()[_layerCount - 1].GetNodes()[0].GetActivation()!;
        _network = network;
    }

    public NetworkManager(string name, ushort layerCount, ushort inputSize,
        ushort hiddenSize, ushort outputSize, Node.ActivationType inputType, Node.ActivationType hiddenType,
        Node.ActivationType outputType, Network.LossFunction lossFunction, double learningRate)
    {
        _name = name;
        _layerCount = layerCount;
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;
        _lossFunction = lossFunction;
        _learningRate = learningRate;
        _inputType = inputType;
        _hiddenType = hiddenType;
        _outputType = outputType;
        _network = GenerateNetwork();
    }

    public Network GenerateNetwork()
    {
        Network network = new Network(_name);
        network.InstantiateBasics(_layerCount - 2, _lossFunction, _learningRate);
        network.CreateInputLayer(_inputSize, _inputType);
        network.CreateHiddenLayers(_hiddenSize, _hiddenType);
        network.CreateOutputLayer(_outputSize, _outputType);
        return network;
    }
    
    public Network FindBest(double[][] features, double[][] outputs, uint range, uint epochs, uint attempts)
    {
        ConcurrentBag<(Network network, double score)> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add((_network, _network.Loss(features, outputs)));
        for (int i = 0; i < cuts; i++)
        {
            Parallel.For(0, coreCount, p =>
            {
                Network network = GenerateNetwork();
                network.Randomize(range);
                NetworkTrainer.Optimize(network, features, outputs, epochs);
                generations.Add((network, network.Loss(features, outputs)));
            });
        }

        return generations.ToList().OrderBy(x => x.score).First().network;
    }

    public double[] GenerateScores(double[][] features, double[][] outputs, uint range, uint epochs, uint attempts)
    {
        ConcurrentBag<double> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add(_network.Loss(features, outputs));
        for (int i = 0; i < cuts; i++)
        {
            Parallel.For(0, coreCount, p =>
            {
                Network network = GenerateNetwork();
                network.Randomize(range);
                NetworkTrainer.Optimize(network, features, outputs, epochs);
                generations.Add(network.Loss(features, outputs));
            });
        }

        //a
        double[] result = generations.ToArray();
        Array.Sort(result);
        return result;
    }
}
//