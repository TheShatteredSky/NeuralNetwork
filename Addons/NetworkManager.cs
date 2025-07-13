namespace NeuralNetwork.Addons;

using System.Collections.Concurrent;
using Core;
using Optimizers;

public class NetworkManager
{
    private string _name;
    private List<Network> _networks;
    private List<(string name, double[][] inputs, double[][] outputs)> _datasets;

    public NetworkManager(String name)
    {
        _name = name;
        _networks = [];
        _datasets = [];
    }

    public void AddNetwork(Network network)
    {
        _networks.Add(network);
    }

    public void RemoveNetwork(int index)
    {
        _networks.RemoveAt(index);
    }

    public void RemoveNetwork(string name)
    {
        _networks.RemoveAt(GetNetworkIndexFromName(name));
    }

    public void RemoveNetwork(Network network)
    {
        _networks.Remove(network);
    }

    public int GetNetworkIndexFromName(string name)
    {
        for (int i = 0; i < _networks.Count; i++)
            if (_networks[i].GetName() == name) return i;
        throw new Exception($"Network {name} not found");
    }
    
    public Network GetNetwork(int index)
    {
        return _networks[index];
    }
    
    public Network GetNetwork(string name)
    {
        return _networks[GetNetworkIndexFromName(name)];
    }

    public bool ContainsNetwork(int index)
    {
        return ContainsNetwork(_networks[index]);
    }

    public bool ContainsNetwork(string name)
    {
        for (int i = 0; i <  _networks.Count; i++)
            if (_networks[i].GetName() == name) return true;
        return false;
    }
    
    public bool ContainsNetwork(Network network)
    {
        return _networks.Contains(network);
    }
    
    public int GetDatasetIndexFromName(string name)
    {
        for (int i = 0; i < _datasets.Count; i++)
            if (_datasets[i].name == name) return i;
        throw new Exception($"Dataset {name} not found");
    }

    public void AddDataset((double[][] inputs, double[][] outputs) data, string name)
    {
        _datasets.Add((name, data.inputs, data.outputs));
    }

    public void RemoveDataset(int index)
    {
        _datasets.RemoveAt(index);
    }

    public void RemoveDataset(string name)
    {
        _datasets.RemoveAt(GetDatasetIndexFromName(name));
    }
    
    public (string name, double[][] inputs, double[][] outputs) GetDataset(int index)
    {
        return _datasets[index];
    }

    public (string name, double[][] inputs, double[][] outputs) GetDataset(string name)
    {
        return _datasets[GetDatasetIndexFromName(name)];
    }

    public Network GenerateNetwork(int i)
    {
        Network original = _networks[i];
        return GenerateNetwork(original);
    }
    
    public Network GenerateNetwork(string name)
    {
        Network original = _networks[GetNetworkIndexFromName(name)];
        return GenerateNetwork(original);
    }
    
    public static Network GenerateNetwork(Network original)
    {
        Network network = new Network(original.GetName());
        network.Instantiate(original.GetLayerCount() - 2);
        for (int l = 0; l < original.GetLayerCount(); l++)
        {
            Layer layer = new Layer((ushort)l, original[l].GetLayerType());
            layer.InstantiateCustom(original[l].GetSize());
            for (int n = 0; n < original[l].GetSize(); n++)
            {
                ushort[] parents = new ushort[original[l, n].GetParentCount()];
                double[] weights = new double[original[l, n].GetDimensions()];
                for (int p = 0; p < original[l, n].GetParentCount(); p++)
                    parents[p] = original[l, n].GetParents()[p];
                for (int w = 0; w < original[l, n].GetDimensions(); w++)
                    weights[w] = original[l, n].GetWeights()[w];
                Node node = new Node((ushort)n, (ushort)l, original[l, n].GetDimensions(), weights, original[l, n].GetBias(), original[l, n].GetActivation(), parents);
                layer[n] = node;
            }
            network[l] = layer;
        }
        return network;
    }
    
    public Network FindBest(int networkIndex, int datasetIndex, Optimizer.LossFunction lossFunction, double learningRate, uint range, uint epochs, uint attempts)
    {
        Network original = _networks[networkIndex];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[datasetIndex];
        return FindBest(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts);
    }

    public Network FindBest(string networkName, string datasetName,  Optimizer.LossFunction lossFunction, double learningRate, uint range, uint epochs, uint attempts)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[GetDatasetIndexFromName(datasetName)];
        return FindBest(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts);
    }
    
    public static Network FindBest(Network original, (double[][] inputs, double[][] outputs) data, Optimizer.LossFunction lossFunction, double learningRate, uint range, uint epochs, uint attempts)
    {
        ConcurrentBag<(Network network, double score)> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add((original, original.Loss(data.inputs, data.outputs)));
        for (int i = 0; i < cuts; i++)
        {
            Parallel.For(0, coreCount, p =>
            {
                Network network = GenerateNetwork(original);
                network.Randomize(range);
                SGDOptimizer optimizer = NetworkTrainer.CreateSGDOptimizer(network, lossFunction, learningRate);
                optimizer.Optimize(data.inputs, data.outputs, epochs);
                generations.Add((network, network.Loss(data.inputs, data.outputs)));
            });
        }
        return generations.ToList().OrderBy(x => x.score).First().network;
    }
    
    public double[] GenerateScores(int networkIndex, int datasetIndex, Optimizer.LossFunction lossFunction, double learningRate, uint range, uint epochs, uint attempts)
    {
        Network original = _networks[networkIndex];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[datasetIndex];
        return GenerateScores(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts);
    }
    
    public double[] GenerateScores(string networkName, string datasetName, Optimizer.LossFunction lossFunction, double learningRate, uint range, uint epochs, uint attempts)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[GetDatasetIndexFromName(datasetName)];
        return GenerateScores(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts);
    }
    
    public static double[] GenerateScores(Network original, (double[][] inputs, double[][] outputs) data, Optimizer.LossFunction lossFunction, double learningRate, uint range, uint epochs, uint attempts)
    {
        ConcurrentBag<double> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add(original.Loss(data.inputs, data.outputs));
        for (int i = 0; i < cuts; i++)
        {
            Parallel.For(0, coreCount, p =>
            {
                Network network = GenerateNetwork(original);
                network.Randomize(range);
                SGDOptimizer optimizer = NetworkTrainer.CreateSGDOptimizer(network, lossFunction, learningRate);
                optimizer.Optimize(data.inputs, data.outputs, epochs);
                generations.Add(network.Loss(data.inputs, data.outputs));
            });
        }
        double[] result = generations.ToArray();
        Array.Sort(result);
        return result;
    }
}