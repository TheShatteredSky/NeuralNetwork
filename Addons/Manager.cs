namespace NeuralNetwork.Addons;

/// <summary>
/// A Manager instance. Made to manage Networks and Datasets
/// </summary>
public class Manager
{
    private List<Network> _networks;
    private List<Dataset> _datasets;
    private bool _log;
    private string _logPath;
    
    /// <summary>
    /// A Manager instance.
    /// </summary>
    public Manager()
    {
        _networks = [];
        _datasets = [];
        _log = false;
        _logPath = "";
    }

    /// <summary>
    /// Start this Manager's logging.
    /// </summary>
    /// <param name="logPath">The file path for the Manager's logs.</param>
    public void StartLogging(string logPath)
    {
        _log = true;
        _logPath = logPath;
    }

    /// <summary>
    /// Starts this Manager's logging. The file path needs to already be set.
    /// </summary>
    /// <exception cref="Exception"></exception>
    public void StartLogging()
    {
        if (_logPath == "") throw new Exception("LogPath not set");
        _log = true;
    }
    
    /// <summary>
    /// Stops this Manager's logging.
    /// </summary>
    public void StopLogging() => _log = false;
    
    /// <summary>
    /// Fetches this Manager's Network array.
    /// </summary>
    /// <returns>This Manager's Network array.</returns>
    public Network[] GetNetworks() => _networks.ToArray();

    /// <summary>
    /// Adds the specified Network to this Manager.
    /// </summary>
    /// <param name="network">The Network to add.</param>
    public void AddNetwork(Network network)
    {
        _networks.Add(network);
    }

    /// <summary>
    /// Removes the Network with the specified name from this Manager.
    /// </summary>
    /// <param name="name">The name of the Network to remove.</param>
    public void RemoveNetwork(string name)
    {
        _networks.RemoveAt(GetNetworkIndexFromName(name));
    }

    /// <summary>
    /// Removes the specified Network from this Manager.
    /// </summary>
    /// <param name="network">The Network to remove.</param>
    public void RemoveNetwork(Network network)
    {
        _networks.Remove(network);
    }
    
    /// <summary>
    /// Gets the index of a Network using its name.
    /// </summary>
    /// <param name="name">The name of the Network.</param>
    /// <returns>The index of the Network.</returns>
    /// <exception cref="Exception"></exception>
    private int GetNetworkIndexFromName(string name)
    {
        for (int i = 0; i < _networks.Count; i++)
            if (_networks[i].GetName() == name) return i;
        throw new Exception($"Network {name} not found.");
    }
    
    /// <summary>
    /// Fetches the Network with the specified name.
    /// </summary>
    /// <param name="name">The name of the Network.</param>
    /// <returns>The Network with the specified name.</returns>
    public Network GetNetwork(string name)
    {
        return _networks[GetNetworkIndexFromName(name)];
    }

    /// <summary>
    /// Checks if this Manager contains a Network with the specified name.
    /// </summary>
    /// <param name="name">The name of the Network.</param>
    /// <returns>True if it contains it, false otherwise.</returns>
    public bool ContainsNetwork(string name)
    {
        foreach (var network in _networks)
            if (network.GetName() == name) return true;
        return false;
    }
    
    /// <summary>
    /// Checks if this Manager contains the specified Network.
    /// </summary>
    /// <param name="network">The Network.</param>
    /// <returns>True if it contains it, false otherwise.</returns>
    public bool ContainsNetwork(Network network)
    {
        return _networks.Contains(network);
    }
    
    /// <summary>
    /// Gets the index of a Dataset using its name.
    /// </summary>
    /// <param name="name">The name of the Dataset.</param>
    /// <returns>The index of the Dataset.</returns>
    /// <exception cref="Exception"></exception>
    private int GetDatasetIndexFromName(string name)
    {
        for (int i = 0; i < _datasets.Count; i++)
            if (_datasets[i].GetName() == name) return i;
        throw new Exception($"Dataset {name} not found.");
    }

    /// <summary>
    /// Adds the specified Dataset to this Manager.
    /// </summary>
    /// <param name="data">The Dataset to add.</param>
    public void AddDataset(Dataset data)
    {
        _datasets.Add(data);
    }

    /// <summary>
    /// Removes the Dataset with the specified name from this Manager.
    /// </summary>
    /// <param name="name">The name of the Dataset to remove.</param>
    public void RemoveDataset(string name)
    {
        _datasets.RemoveAt(GetDatasetIndexFromName(name));
    }

    /// <summary>
    /// Removes the specified Dataset from this Manager.
    /// </summary>
    /// <param name="data">The Dataset to remove.</param>
    public void RemoveDataset(Dataset data)
    {
        _datasets.Remove(data);
    }

    /// <summary>
    /// Fetches the Dataset with the specified name.
    /// </summary>
    /// <param name="name">The name of the Dataset.</param>
    /// <returns>The Dataset with the specified name.</returns>
    public Dataset GetDataset(string name)
    {
        return _datasets[GetDatasetIndexFromName(name)];
    }
    
    /// <summary>
    /// Generates a deep copy of the Network with the specified name.
    /// </summary>
    /// <param name="name">The name of the Network to copy.</param>
    /// <returns>A deep copy of the Network.</returns>
    private Network GenerateNetwork(string name)
    {
        Network original = _networks[GetNetworkIndexFromName(name)];
        return GenerateNetwork(original);
    }
    
    /// <summary>
    /// Generates a deep copy of the specified Network.
    /// </summary>
    /// <param name="network">The Network to copy.</param>
    /// <returns>A deep copy of the Network.</returns>
    private static Network GenerateNetwork(Network network)
    {
        return network.Clone();
    }

    /// <summary>
    /// Optimizes the Network with the specified name using the Dataset with the specified name.
    /// </summary>
    /// <param name="networkName">The name of the Network to optimize.</param>
    /// <param name="datasetName">The name of the Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    public void Optimize(string networkName, string datasetName, LossType lossFunction, double learningRate,uint epochs, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        Dataset data = _datasets[GetDatasetIndexFromName(datasetName)];
        Optimize(original, data, lossFunction, learningRate, epochs, optimizerType);
    }
    
    /// <summary>
    /// Optimizes the specified Network using the specified Dataset
    /// </summary>
    /// <param name="network">The Network to optimize.</param>
    /// <param name="data">The Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    public static void Optimize(Network network, Dataset data, LossType lossFunction, double learningRate, uint epochs, OptimizerType optimizerType)
    {
        switch (optimizerType)
        {
            case OptimizerType.SGD:
                SGDOptimizer optimizer = new SGDOptimizer(network, lossFunction, learningRate);
                optimizer.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                break;
            case OptimizerType.Adam:
                AdamOptimizer adam = new AdamOptimizer(network, lossFunction, learningRate);
                adam.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                break;
        }
    }

    /// <summary>
    /// Finds the best optimized alternate variant of the Network with the specified name using the Dataset with the specified name.
    /// </summary>
    /// <param name="networkName">The name of the Network to optimize.</param>
    /// <param name="datasetName">The name of the Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="range">The range for randomization.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="attempts">The number of generation attempts.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    /// <returns>The best alternate variant of the Network with the specified name.</returns>
    public Network FindBest(string networkName, string datasetName, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        Dataset data = _datasets[GetDatasetIndexFromName(datasetName)];
        return FindBest(original, data, lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    /// <summary>
    /// Find the best optimized alternate variant of the specified Network using the specified Dataset.
    /// </summary>
    /// <param name="network">The Network to optimize.</param>
    /// <param name="data">The Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="range">The range for randomization.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="attempts">The number of generation attempts.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    /// <returns>The best alternate variant of the Network.</returns>
    //TODO: Make this store the setting arrays of the networks instead of the networks themselves to save memory.
    public static Network FindBest(Network network, Dataset data, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        ConcurrentBag<(Network, double)> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add((network, network.Loss(data.GetInputs(), data.GetOutputs(), lossFunction)));
        switch (optimizerType)
        {
            case OptimizerType.SGD:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network attempt = GenerateNetwork(network);
                        attempt.Randomize(-range, range);
                        SGDOptimizer optimizer = new SGDOptimizer(attempt, lossFunction, learningRate);
                        optimizer.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                        generations.Add((attempt, attempt.Loss(data.GetInputs(), data.GetOutputs(), lossFunction)));
                    });
                }
                break;
            case OptimizerType.Adam:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network attempt = GenerateNetwork(network);
                        attempt.Randomize(-range, range);
                        AdamOptimizer optimizer = new AdamOptimizer(attempt, lossFunction, learningRate);
                        optimizer.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                        generations.Add((attempt, attempt.Loss(data.GetInputs(), data.GetOutputs(), lossFunction)));
                    });
                }
                break;
        }
        return generations.ToArray().OrderBy(x => x.Item2).First().Item1;
    }
    
    /// <summary>
    /// Generates and optimizes alternates of the Network with the specified name using the Dataset with the specified name.
    /// </summary>
    /// <param name="networkName">The name of the Network to optimize.</param>
    /// <param name="datasetName">The name of the Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="range">The range for randomization.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="attempts">The number of generation attempts.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    /// <returns>The optimized alternates of the Network with the specified name.</returns>
    public Network[] GenerateAndOptimizeAlternates(string networkName, string datasetName, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        Dataset data = _datasets[GetDatasetIndexFromName(datasetName)];
        return GenerateAndOptimizerAlternates(original, data, lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    /// <summary>
    /// Generates and optimizes alternates of the specified Network using the specified Dataset.
    /// </summary>
    /// <param name="network">The Network to optimize.</param>
    /// <param name="data">The Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="range">The range for randomization.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="attempts">The number of generation attempts.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    /// <returns>The optimized alternates of the specified Network.</returns>
    public static Network[] GenerateAndOptimizerAlternates(Network network, Dataset data, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
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
                        Network attempt = GenerateNetwork(network);
                        attempt.Randomize(-range, range);
                        SGDOptimizer optimizer = new SGDOptimizer(attempt, lossFunction, learningRate);
                        optimizer.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                        generations.Add(attempt);
                    });
                }
                break;
            case OptimizerType.Adam:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network attempt = GenerateNetwork(network);
                        attempt.Randomize(-range, range);
                        AdamOptimizer optimizer = new AdamOptimizer(attempt, lossFunction, learningRate);
                        optimizer.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                        generations.Add(attempt);
                    });
                }
                break;
        }
        return generations.ToArray();
    }
    
    /// <summary>
    /// Generates the losses of the optimized alternates of the Network with the specified name using the Dataset with the specified name.
    /// </summary>
    /// <param name="networkName">The name of the Network to optimize.</param>
    /// <param name="datasetName">The name of the Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="range">The range for randomization.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="attempts">The number of generation attempts.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    /// <returns>The losses of the optimized alternates of the Network with the specified name.</returns>
    public double[] GenerateLosses(string networkName, string datasetName, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        Network original = _networks[GetNetworkIndexFromName(networkName)];
        Dataset data = _datasets[GetDatasetIndexFromName(datasetName)];
        return GenerateLosses(original, data, lossFunction, learningRate, range, epochs, attempts, optimizerType);
    }
    
    /// <summary>
    /// Generates the losses of the optimized alternates of the specified Network using the specified Dataset.
    /// </summary>
    /// <param name="network">The Network to optimize.</param>
    /// <param name="data">The Dataset to utilize.</param>
    /// <param name="lossFunction">The loss function for predictions.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="range">The range for randomization.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="attempts">The number of generation attempts.</param>
    /// <param name="optimizerType">The type of Optimizer to use.</param>
    /// <returns>The losses of the optimized alternates of the specified Network.</returns>
    public static double[] GenerateLosses(Network network, Dataset data, LossType lossFunction, double learningRate, uint range, uint epochs, uint attempts, OptimizerType optimizerType)
    {
        ConcurrentBag<double> generations = new();
        int coreCount = Environment.ProcessorCount;
        int cuts = (int)attempts / coreCount;
        generations.Add(network.Loss(data.GetInputs(), data.GetOutputs(), lossFunction));
        switch (optimizerType)
        {
            case OptimizerType.SGD:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network attempt = GenerateNetwork(network);
                        attempt.Randomize(-range, range);
                        SGDOptimizer optimizer = new SGDOptimizer(attempt, lossFunction, learningRate);
                        optimizer.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                        generations.Add(attempt.Loss(data.GetInputs(), data.GetOutputs(), lossFunction));
                    });
                }
                break;
            case OptimizerType.Adam:
                for (int i = 0; i < cuts; i++)
                {
                    Parallel.For(0, coreCount, _ =>
                    {
                        Network attempt = GenerateNetwork(network);
                        attempt.Randomize(-range, range);
                        AdamOptimizer optimizer = new AdamOptimizer(attempt, lossFunction, learningRate);
                        optimizer.Optimize(data.GetInputs(), data.GetOutputs(), epochs);
                        generations.Add(attempt.Loss(data.GetInputs(), data.GetOutputs(), lossFunction));
                    });
                }
                break;
        }
        double[] result = generations.ToArray();
        Array.Sort(result);
        return result;
    }
}