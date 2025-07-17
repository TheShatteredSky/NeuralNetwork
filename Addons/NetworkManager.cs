namespace NeuralNetwork.Addons;

public class NetworkManager
{
    private string _name;
    private List<Network> _networks;
    private List<(string name, double[][] inputs, double[][] outputs)> _datasets;
    private bool _log;
    private string _logPath;
    
    public NetworkManager()
    {
        _name = "#";
        _networks = [];
        _datasets = [];
        _log = false;
        _logPath = "";
    }

    public void StartLogging(string logPath)
    {
        _log = true;
        _logPath = logPath;
    }

    public void StartLogging()
    {
        if (_logPath == "") throw new Exception("LogPath not set");
        _log = true;
    }
    
    public void StopLogging() => _log = false;
    public void SetName(string name) => _name = name;
    public string GetName() =>  _name;
    
    public Network[] GetNetworks() => _networks.ToArray();

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

    private int GetNetworkIndexFromName(string name)
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
        foreach (var network in _networks)
            if (network.GetName() == name) return true;
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
        Network network = new Network();
        network.SetName(original.GetName());
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
    
    public void Optimize(int networkIndex, int datasetIndex, LossType lossFunction, double learningRate, uint epochs, OptimizerType optimizerType)
    {
        Network original = _networks[networkIndex];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[datasetIndex];
        Optimize(original, (data.inputs, data.outputs), lossFunction, learningRate, epochs, optimizerType);
    }

    public void Optimize(string networkName, string datasetName, LossType lossFunction, double learningRate,uint epochs, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[GetDatasetIndexFromName(datasetName)];
        Optimize(original, (data.inputs, data.outputs), lossFunction, learningRate, epochs, optimizerType);
    }
    
    public static void Optimize(Network original, (double[][] inputs, double[][] outputs) data, LossType lossFunction, double learningRate, uint epochs, OptimizerType optimizerType)
    {
        switch (optimizerType)
        {
            case OptimizerType.SGD:
                SGDOptimizer optimizer = new SGDOptimizer(original, lossFunction, learningRate);
                optimizer.Optimize(data.inputs, data.outputs, epochs);
                break;
        }
    }
    
    public Network FindBest(int networkIndex, int datasetIndex, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[networkIndex];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[datasetIndex];
        return FindBest(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }

    public Network FindBest(string networkName, string datasetName, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[GetDatasetIndexFromName(datasetName)];
        return FindBest(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    public static Network FindBest(Network original, (double[][] inputs, double[][] outputs) data, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        ConcurrentBag<(Network network, double score)> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add((original, original.Loss(data.inputs, data.outputs, lossFunction)));
        switch (optimizerType)
        {
            case OptimizerType.SGD:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network network = GenerateNetwork(original);
                        network.Randomize(range);
                        SGDOptimizer optimizer = new SGDOptimizer(network, lossFunction, learningRate);
                        optimizer.Optimize(data.inputs, data.outputs, epochs);
                        generations.Add((network, network.Loss(data.inputs, data.outputs, lossFunction)));
                    });
                }
                break;
        }

        var ordered = generations.ToList().OrderBy(x => x.score);
        var result = ordered.First();
        return result.network;
    }
    
    public Network[] GenerateAlternates(int networkIndex, int datasetIndex, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[networkIndex];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[datasetIndex];
        return GenerateAlternates(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    public Network[] GenerateAlternates(string networkName, string datasetName, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[GetDatasetIndexFromName(datasetName)];
        return GenerateAlternates(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    public static Network[] GenerateAlternates(Network original, (double[][] inputs, double[][] outputs) data, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        ConcurrentBag<Network> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        switch (optimizerType)
        {
            case OptimizerType.SGD:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network network = GenerateNetwork(original);
                        network.Randomize(range);
                        SGDOptimizer optimizer = new SGDOptimizer(network, lossFunction, learningRate);
                        optimizer.Optimize(data.inputs, data.outputs, epochs);
                        generations.Add(network);
                    });
                }
                break;
        }
        return generations.ToArray();
    }
    
    public double[] GenerateScores(int networkIndex, int datasetIndex, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[networkIndex];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[datasetIndex];
        return GenerateScores(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    public double[] GenerateScores(string networkName, string datasetName, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        (string name, double[][] inputs, double[][] outputs) data = _datasets[GetDatasetIndexFromName(datasetName)];
        return GenerateScores(original, (data.inputs, data.outputs), lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    public static double[] GenerateScores(Network original, (double[][] inputs, double[][] outputs) data, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        ConcurrentBag<double> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add(original.Loss(data.inputs, data.outputs, lossFunction));
        switch (optimizerType)
        {
            case OptimizerType.SGD:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network network = GenerateNetwork(original);
                        network.Randomize(range);
                        SGDOptimizer optimizer = new SGDOptimizer(network, lossFunction, learningRate);
                        optimizer.Optimize(data.inputs, data.outputs, epochs);
                        generations.Add(network.Loss(data.inputs, data.outputs, lossFunction));
                    });
                }
                break;
        }
        double[] result = generations.ToArray();
        Array.Sort(result);
        return result;
    }
}