namespace NeuralNetwork.Core;

/// <summary>
/// A Network instance.
/// </summary>
public class Network
{
    private string? _name;
    private ushort _layerCount;
    private Layer[] _networkLayers;
    private (double shift, double scale, double deshift)[]? _inputScaling;
    private (double shift, double scale, double deshift)[]? _outputScaling;

    /// <summary>
    /// Completely empty constructor if needed.
    /// </summary>
    public Network()
    {
        _name = null;
        _layerCount = 0;
        _networkLayers = [];
        _inputScaling = null;
        _outputScaling = null;
    }

    /// <summary>
    /// Creates a new Network with the specified name.
    /// </summary>
    /// <param name="name">The name of the Network.</param>
    public Network(string name)
    {
        _name = name;
        _layerCount = 0;
        _networkLayers = [];
        _inputScaling = null;
        _outputScaling = null;
    }

    /// <summary>
    /// Creates the Layer array of the Network (although empty).
    /// </summary>
    /// <param name="hiddenLayers">The number of hidden Layers, the total number of Layers will be that +2.</param>
    public void Instantiate(int hiddenLayers)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
    }

    /// <summary>
    /// Creates the Layer array of the Network and instantiates them.
    /// </summary>
    /// <param name="inputSize">The number of input Nodes.</param>
    /// <param name="inputActivation">The activation for the input Nodes.</param>
    /// <param name="hiddenLayers">The number of hidden Layers.</param>
    /// <param name="hiddenSize">The number of hidden Nodes in each hidden Layer.</param>
    /// <param name="hiddenActivation">The activation for the hidden Nodes.</param>
    /// <param name="outputSize">The number of output Nodes.</param>
    /// <param name="outputActivation">The activation for the output Nodes.</param>
    public void Instantiate(ushort inputSize, ActivationType inputActivation, int hiddenLayers, ushort hiddenSize, ActivationType hiddenActivation, ushort outputSize, ActivationType outputActivation)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
        CreateInputLayer(inputSize, inputActivation);
        CreateHiddenLayers(hiddenSize, hiddenActivation);
        CreateOutputLayer(outputSize, outputActivation);
    }
    
    /// <summary>
    /// Sets the name of the Network.
    /// </summary>
    /// <param name="name">The new name of the Network</param>
    public void SetName(string name) => _name = name;
    
    /// <summary>
    /// Sets the input scaling of the Network.
    /// </summary>
    /// <param name="scales">The new input scales.</param>
    public void SetInputScaling((double, double, double)[] scales) => _inputScaling = scales;
    
    /// <summary>
    /// Sets the output scaling of the Network.
    /// </summary>
    /// <param name="scales">The new output scales.</param>
    public void SetOutputScaling((double, double, double)[] scales) => _outputScaling = scales;
    
    /// <summary>
    /// Fetches the name of the Network.
    /// </summary>
    /// <returns>The name of the Network.</returns>
    public void GetName(string newName) => _name = newName;

    public string GetName() => _name!;
    
    /// <summary>
    /// Fetches the Layer count of the Network.
    /// </summary>
    /// <returns>The number of layers in the network.</returns>
    public ushort GetLayerCount() => _layerCount;
    
    /// <summary>
    /// Fetches to Layer array of the network.
    /// ⚠ This returns a reference to the actual array the Network uses, modifying it will modify the that of Network too.
    /// </summary>
    /// <returns>The Layer array of the Network.
    /// </returns>
    public Layer[] GetLayers() => _networkLayers;

    /// <summary>
    /// Indexer for the network's Layers.
    /// ⚠ This returns a reference to the actual Layer the Network uses, modifying it will modify the that of Network too.
    /// </summary>
    /// <param name="layer">The index of the Layer.</param>
    public Layer this[int layer]
    {
        get => _networkLayers[layer];
        set => _networkLayers[layer] = value;
    }
    
    /// <summary>
    /// Indexer for the Network's Nodes.
    /// ⚠ This returns a reference to the actual Node the Network uses, modifying it will modify the that of Network too.
    /// </summary>
    /// <param name="layer">The index of the Layer.</param>
    /// <param name="node">The index of the Node.</param>
    public Node this[int layer, int node]
    {
        get => this[layer][node];
        set => this[layer][node] = value;
    }
    
    /// <summary>
    /// Indexer for the Network's parameters.
    /// </summary>
    /// <param name="layer">The index of the Layer.</param>
    /// <param name="node">The index of the Node.</param>
    /// <param name="param">The index of the parameter, 0 to dimensions - 1 will return the specified weight, while dimensions will return the bias.</param>
    public double this[int layer, int node, int param]
    {
        get => this[layer, node][param];
        set => this[layer, node][param] = value;
    }

    /// <summary>
    /// Creates the input Layer of the Network.
    /// </summary>
    /// <param name="numberOfNodes">The number of Nodes in the input Layer.</param>
    /// <param name="activation">The activation for the input Nodes.</param>
    /// <exception cref="ArgumentException"></exception>
    public void CreateInputLayer(ushort numberOfNodes, ActivationType activation)
    {
        if (activation == ActivationType.Softmax) throw new ArgumentException("Input layers cannot have a Softmax activation.");
        Layer input = new Layer(0, LayerType.Input);
        input.Instantiate(numberOfNodes, 1, activation);
        _networkLayers[0] = input;
    }
    
    /// <summary>
    /// Creates the hidden Layers of the Network.
    /// </summary>
    /// <param name="numberOfNodes">The number of Nodes in each hidden Layer.</param>
    /// <param name="activation">The activation for the hidden Nodes.</param>
    /// <exception cref="ArgumentException"></exception>
    public void CreateHiddenLayers(ushort numberOfNodes, ActivationType activation)
    {
        if (activation == ActivationType.Softmax) throw new ArgumentException("Hidden layers cannot have a Softmax activation.");
        Layer afterInputLayer = new Layer(1, LayerType.Hidden);
        afterInputLayer.Instantiate(numberOfNodes, _networkLayers[0].GetSize(), activation);
        _networkLayers[1] = afterInputLayer;
        for (int i = 2; i < _layerCount - 1; i++)
        {
            Layer hiddenLayer = new Layer((ushort)i, LayerType.Hidden);
            hiddenLayer.Instantiate(numberOfNodes, numberOfNodes, activation);
            _networkLayers[i] = hiddenLayer;
        }
    }

    /// <summary>
    /// Creates the output Layer of the Network.
    /// </summary>
    /// <param name="numberOfNodes">The number of Nodes in the output Layer.</param>
    /// <param name="activation">The activation for the output Nodes.</param>
    /// <exception cref="ArgumentException"></exception>
    public void CreateOutputLayer(ushort numberOfNodes, ActivationType activation)
    {
        Layer output = new Layer((ushort)(_layerCount - 1)!, LayerType.Output);
        output.Instantiate(numberOfNodes, _networkLayers[_layerCount - 2].GetSize(), activation);
        _networkLayers[_layerCount - 1] = output;
    }

    /// <summary>
    /// Unscales (morphs to inner range) inputs and outputs.
    /// </summary>
    /// <param name="inputs">The inputs of the data.</param>
    /// <param name="outputs">The outputs of the data.</param>
    /// <returns>The unscaled inputs and outputs.</returns>
    public (double[][] unscaledInputs, double[][] unscaledOuputs) UnscaledData(double[][] inputs, double[][] outputs)
    {
        (double[][] unscaledInputs, double[][] unscaledOuputs) unscaledData = new();
        unscaledData.unscaledInputs = new double[inputs.Length][];
        unscaledData.unscaledOuputs = new double[outputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            unscaledData.unscaledInputs[i] = UnscaledInputs(inputs[i]);
        for (int i = 0; i < outputs.Length; i++)
            unscaledData.unscaledOuputs[i] = UnscaledOutputs(outputs[i]);
        return unscaledData;
    }
    
    /// <summary>
    /// Scales (morphs to outer range) inputs and outputs. 
    /// </summary>
    /// <param name="inputs">The inputs of the data.</param>
    /// <param name="outputs">The outputs of the data.</param>
    /// <returns>The scaled inputs and outputs.</returns>
    public (double[][] scaledInputs, double[][] scaledOuputs) ScaledData(double[][] inputs, double[][] outputs)
    {
        (double[][] scaledInputs, double[][] scaledOuputs) scaledData = new();
        scaledData.scaledInputs = new double[inputs.Length][];
        scaledData.scaledOuputs = new double[outputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            scaledData.scaledInputs[i] = ScaledInputs(inputs[i]);
        for (int i = 0; i < outputs.Length; i++)
            scaledData.scaledOuputs[i] = ScaledOutputs(outputs[i]);
        return scaledData;
    }

    /// <summary>
    /// Scales (morphs to inner range) inputs of data.
    /// </summary>
    /// <param name="inputs">The data to unscale.</param>
    /// <returns>The unscaled inputs.</returns>
    public double[] UnscaledInputs(double[] inputs)
    {
        if (_inputScaling == null) return inputs;
        for (int i = 0; i < inputs.Length; i++)
            inputs[i] = (inputs[i] + _inputScaling[i].shift)  * _inputScaling[i].scale + _inputScaling[i].deshift;
        return inputs;
    }
    
    /// <summary>
    /// Scales (morphs to outer range) inputs of data.
    /// </summary>
    /// <param name="inputs">The data to scale.</param>
    /// <returns>The scaled inputs.</returns>
    public double[] ScaledInputs(double[] inputs)
    {
        if (_inputScaling == null) return inputs;
        for (int i = 0; i < inputs.Length; i++)
            inputs[i] = (inputs[i] - _inputScaling[i].deshift) / _inputScaling[i].scale - _inputScaling[i].shift;
        return inputs;
    }
    
    /// <summary>
    /// Scales (morphs to inner range) outputs of data.
    /// </summary>
    /// <param name="outputs">The data to unscale.</param>
    /// <returns>The unscaled outputs.</returns>
    public double[] UnscaledOutputs(double[] outputs)
    {
        if (_outputScaling == null) return outputs;
        for (int i = 0; i < outputs.Length; i++)
            outputs[i] = (outputs[i] + _outputScaling[i].shift)  * _outputScaling[i].scale + _outputScaling[i].deshift;
        return outputs;
    }
    
    /// <summary>
    /// Scales (morphs to outer range) outputs of data.
    /// </summary>
    /// <param name="outputs">The data to scale.</param>
    /// <returns>The scaled outputs.</returns>
    public double[] ScaledOutputs(double[] outputs)
    {
        if (_outputScaling == null) return outputs;
        for (int i = 0; i < outputs.Length; i++)
            outputs[i] = (outputs[i] - _outputScaling[i].deshift) / _outputScaling[i].scale - _outputScaling[i].shift;
        return outputs;
    }
    
    /// <summary>
    /// Base process method for the Network.
    /// </summary>
    /// <param name="inputs">The data to process.</param>
    /// <returns>The predictions of the Network for each data instance.</returns>
    /// <exception cref="ArgumentException"></exception>
    //TODO: Implemented parallel processing for this method. Will need to check that no issues arise.
    public double[][] Process(double[][] inputs)
    {
        ConcurrentBag<double[]> outputs = new ConcurrentBag<double[]>();
        Parallel.For(0, inputs.Length, i =>
        {
            if (inputs[i].Length != this[0].GetSize()) throw new ArgumentException($"Number of inputs does not match the size of the input layer. (Sample #{i})");
            outputs.Add(ProcessSingle(inputs[i]));
        });
        return outputs.ToArray();
    }
    
    /// <summary>
    /// Processes a single instance of data.
    /// </summary>
    /// <param name="inputs">The data to process.</param>
    /// <returns>The predictions of the Network.</returns>
    /// <exception cref="ArgumentException"></exception>
    internal double[] ProcessSingle(double[] inputs)
    {
        if (inputs.Length != this[0].GetSize()) throw new ArgumentException("Number of inputs does not match the size of the input layer.");
        double[] current = UnscaledInputs(inputs);
        foreach (var layer in _networkLayers)
            current = layer.Process(current);
        return ScaledOutputs(current);
    }

    /// <summary>
    /// Computes the loss of the Network.
    /// </summary>
    /// <param name="inputs">The inputs of data.</param>
    /// <param name="outputs">The outputs of data.</param>
    /// <param name="lossType">The loss function.</param>
    /// <returns>The loss value of the Network on the specified data.</returns>
    /// <exception cref="ArgumentException"></exception>
    public double Loss(double[][] inputs, double[][] outputs, LossType lossType)
    {
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this network generates. (Sample #{i})");
            double[] scaledPredictions = ProcessSingle(inputs[i]);
            double[] unscaledPredictions = UnscaledOutputs(scaledPredictions);
            double[] unscaledOutputs = UnscaledOutputs(outputs[i]);
            double sampleLoss = 0;
            switch (lossType)
            {
                case LossType.MSE:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.MSE(unscaledPredictions[j], unscaledOutputs[j]);
                    sampleLoss /= outputs[i].Length;
                    break;
                case LossType.BinaryCrossEntropy:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.BinaryCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                    break;
                case LossType.CategoricalCrossEntropy:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.CategoricalCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                    break;
            }
            totalError += sampleLoss;
        }

        return totalError / inputs.Length;
    }
    
    /// <summary>
    /// Computes the loss of the Network with insights on the predictions.
    /// </summary>
    /// <param name="inputs">The inputs of data.</param>
    /// <param name="outputs">The outputs of data.</param>
    /// <param name="lossFunction">The loss function.</param>
    /// <returns>A string with the loss value and calculation insights.</returns>
    /// <exception cref="ArgumentException"></exception>
    public string LossString(double[][] inputs, double[][] outputs, LossType lossFunction)
    {
        StringBuilder sb = new StringBuilder();
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this network generates. (Sample #{i})");
            double currError = 0;
            double[] scaledPredictions = ProcessSingle(inputs[i]);
            double[] unscaledPredictions = UnscaledOutputs(scaledPredictions);
            double[] scaledOutputs = outputs[i];
            double[] unscaledOutputs = UnscaledOutputs(outputs[i]);
            for (int j = 0; j < outputs[i].Length; j++)
            {
                switch (lossFunction)
                {
                    case LossType.MSE:
                        currError += LossFunction.MSE(unscaledPredictions[j], unscaledOutputs[j]);
                        currError /= outputs[i].Length;
                        break;
                    case LossType.BinaryCrossEntropy:
                        currError += LossFunction.BinaryCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                        break;
                    case LossType.CategoricalCrossEntropy:
                        currError += LossFunction.CategoricalCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                        break;
                }
            }
            totalError += currError;
            sb.AppendLine($"#{i} Input: {string.Join(", ", inputs[i])} Predicted: {string.Join(", ", scaledPredictions)} Expected: {string.Join(", ", scaledOutputs)} Loss: {currError} Total: {totalError}");
        }
        totalError /= inputs.Length;
        sb.AppendLine($"Loss: {totalError}");
        return sb.ToString();
    }
    
    /// <summary>
    /// The formatted string of the Network.
    /// ⚠ This is a custom format, not JSON.
    /// </summary>
    /// <returns>A string representing the network.</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append($"{_name};{_layerCount}\n");
        foreach (Layer layer in _networkLayers)
            sb.Append(layer);
        sb.AppendLine("#SCALES");
        if (_inputScaling == null) sb.AppendLine("#");
        else for (int i = 0; i < _inputScaling.Length; i++)
            sb.AppendLine(_inputScaling[i].shift + "," + _inputScaling[i].scale + "," + _inputScaling[i].deshift + (i == _outputScaling.Length - 1 ? "" : ";"));
        if (_outputScaling == null) sb.AppendLine("#");
        else for (int i = 0; i < _outputScaling.Length; i++)
            sb.Append(_outputScaling[i].shift + "," + _outputScaling[i].scale + "," + _outputScaling[i].deshift + (i == _outputScaling.Length - 1 ? "" : ";"));
        return sb.ToString();
    }

    /// <summary>
    /// Randomizes all parameters of the Network in the specified range.
    /// </summary>
    /// <param name="min">The minimum of the range.</param>
    /// <param name ="max">The maximum of the range.</param>
    public void Randomize(double min, double max)
    {
        foreach (var layer in _networkLayers)
            foreach (var node in layer.GetNodes())
                for (int i = 0; i <= node.GetDimensions(); i++)
                    node[i] = NetworkUtilities.NextDouble(min, max);
    }

    /// <summary>
    /// Clones the Network.
    /// </summary>
    /// <returns>A copy of the Network.</returns>
    //TODO: Check this actually works.
    public Network Clone()
    {
        Network network = new Network();
        if (_name != null) network.SetName(_name);
        network.Instantiate(this[0].GetSize(), this[0, 0].GetActivation(), _layerCount - 2,this[1].GetSize(), this[1, 0].GetActivation(), this[_layerCount - 1].GetSize(), this[_layerCount - 1, 0].GetActivation());
        if (_inputScaling != null) network.SetInputScaling(_inputScaling);
        if (_outputScaling != null) network.SetOutputScaling(_outputScaling);
        for (int l = 0; l < _layerCount; l++)
        {
            for (int n = 0; n < this[l].GetSize(); n++)
            {
                ushort[] parents = new ushort[this[l, n].GetParentCount()];
                double[] weights = new double[this[l, n].GetDimensions()];
                for (int p = 0; p < this[l, n].GetParentCount(); p++)
                    parents[p] = this[l, n].GetParents()[p];
                for (int w = 0; w < this[l, n].GetDimensions(); w++)
                    weights[w] = this[l, n].GetWeights()[w];
                network[l, n].SetParents(parents);
                network[l, n].SetWeights(weights);
                network[l, n].SetBias(this[l, n].GetBias());
            }
        }
        return network;
    }
}