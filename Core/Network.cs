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
    /// Indexer for this Network's Layers.
    /// ⚠ This method returns a reference to the actual Layer this Network uses, modifying it will modify the that of this Network too.
    /// </summary>
    /// <param name="layerIndex">The index of the Layer.</param>
    public Layer this[int layerIndex]
    {
        get => _networkLayers[layerIndex];
        set => _networkLayers[layerIndex] = value;
    }
    
    /// <summary>
    /// Indexer for this Network's Nodes.
    /// ⚠ This method returns a reference to the actual Node this Network uses, modifying it will modify the that of this Network too.
    /// </summary>
    /// <param name="layerIndex">The index of the Layer.</param>
    /// <param name="nodeIndex">The index of the Node.</param>
    public Node this[int layerIndex, int nodeIndex]
    {
        get => this[layerIndex][nodeIndex];
        set => this[layerIndex][nodeIndex] = value;
    }
    
    /// <summary>
    /// Indexer for this Network's parameters.
    /// </summary>
    /// <param name="layerIndex">The index of the Layer.</param>
    /// <param name="nodeIndex">The index of the Node.</param>
    /// <param name="param">The index of the parameter, 0 to dimensions - 1 will return the specified weight, while dimensions will return the bias.</param>
    public double this[int layerIndex, int nodeIndex, int param]
    {
        get => this[layerIndex, nodeIndex][param];
        set => this[layerIndex, nodeIndex][param] = value;
    }
    
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
    /// <param name="name">The name of this Network.</param>
    public Network(string name)
    {
        _name = name;
        _layerCount = 0;
        _networkLayers = [];
        _inputScaling = null;
        _outputScaling = null;
    }

    /// <summary>
    /// Creates the Layer array of this Network (although empty).
    /// </summary>
    /// <param name="hiddenLayers">The number of hidden Layers, the total number of Layers will be that +2.</param>
    public void Instantiate(int hiddenLayers)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
    }

    /// <summary>
    /// Creates the Layer array of this Network and instantiates them.
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
        InstantiateInputLayer(inputSize, inputActivation);
        InstantiateHiddenLayers(hiddenSize, hiddenActivation);
        InstantiateOutputLayer(outputSize, outputActivation);
    }
    
    /// <summary>
    /// Fetches the name of this Network.
    /// </summary>
    /// <returns>This Network's name.</returns>
    public string? GetName() => _name;
    
    /// <summary>
    /// Sets the name of this Network.
    /// </summary>
    /// <param name="name">The new name of this Network</param>
    public void SetName(string? name) => _name = name;
    
    /// <summary>
    /// Sets the input scaling of this Network.
    /// </summary>
    /// <param name="scales">The new input scales.</param>
    public void SetInputScaling((double, double, double)[]? scales) => _inputScaling = scales == null ? null : Utilities.CopyNonObjectArray(scales);
    
    /// <summary>
    /// Sets the output scaling of this Network.
    /// </summary>
    /// <param name="scales">The new output scales.</param>
    public void SetOutputScaling((double, double, double)[]? scales) => _outputScaling = scales == null ? null : Utilities.CopyNonObjectArray(scales);
    
    /// <summary>
    /// Fetches the Layer count of this Network.
    /// </summary>
    /// <returns>This Network's Layer count.</returns>
    public ushort GetLayerCount() => _layerCount;
    
    /// <summary>
    /// Fetches the Layer array of this Network.
    /// </summary>
    /// <returns>This Network's Layer array./// </returns>
    public IReadOnlyList<Layer> GetLayers() => Utilities.ConvertToReadOnlyList(_networkLayers);

    /// <summary>
    /// Instantiates the input Layer of this Network.
    /// </summary>
    /// <param name="numberOfNodes">The number of Nodes in the input Layer.</param>
    /// <param name="activation">The activation for the input Nodes.</param>
    /// <exception cref="ArgumentException"></exception>
    public void InstantiateInputLayer(ushort numberOfNodes, ActivationType activation)
    {
        if (activation == ActivationType.Softmax) throw new ArgumentException("Input Layers cannot have a Softmax activation.");
        Layer input = new Layer(LayerType.Input);
        input.Instantiate(numberOfNodes, 1, activation);
        _networkLayers[0] = input;
    }
    
    /// <summary>
    /// Instantiates the hidden Layers of this Network.
    /// </summary>
    /// <param name="numberOfNodes">The number of Nodes in each hidden Layer.</param>
    /// <param name="activation">The activation for the hidden Nodes.</param>
    /// <exception cref="ArgumentException"></exception>
    public void InstantiateHiddenLayers(ushort numberOfNodes, ActivationType activation)
    {
        if (activation == ActivationType.Softmax) throw new ArgumentException("Hidden Layers cannot have a Softmax activation.");
        Layer afterInputLayer = new Layer(LayerType.Hidden);
        afterInputLayer.Instantiate(numberOfNodes, _networkLayers[0].GetSize(), activation);
        _networkLayers[1] = afterInputLayer;
        for (int i = 2; i < _layerCount - 1; i++)
        {
            Layer hiddenLayer = new Layer(LayerType.Hidden);
            hiddenLayer.Instantiate(numberOfNodes, numberOfNodes, activation);
            _networkLayers[i] = hiddenLayer;
        }
    }

    /// <summary>
    /// Instantiates the output Layer of this Network.
    /// </summary>
    /// <param name="numberOfNodes">The number of Nodes in the output Layer.</param>
    /// <param name="activation">The activation for the output Nodes.</param>
    /// <exception cref="ArgumentException"></exception>
    public void InstantiateOutputLayer(ushort numberOfNodes, ActivationType activation)
    {
        Layer output = new Layer(LayerType.Output);
        output.Instantiate(numberOfNodes, _networkLayers[_layerCount - 2].GetSize(), activation);
        _networkLayers[_layerCount - 1] = output;
    }

    /// <summary>
    /// Unscales (morphs to inner range) inputs and outputs.
    /// </summary>
    /// <param name="inputs">The inputs of the data.</param>
    /// <param name="outputs">The outputs of the data.</param>
    /// <returns>The unscaled inputs and outputs.</returns>
    //Note: At no point should the input or output array be modified or returned.
    public (double[][] unscaledInputs, double[][] unscaledOuputs) UnscaledData(double[][] inputs, double[][] outputs)
    {
        (double[][] unscaledInputs, double[][] unscaledOuputs) unscaledData = (new double[inputs.Length][],  new double[outputs.Length][]);
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
    //Note: At no point should the input or output array be modified or returned.
    public (double[][] scaledInputs, double[][] scaledOuputs) ScaledData(double[][] inputs, double[][] outputs)
    {
        (double[][] scaledInputs, double[][] scaledOuputs) scaledData = (new double[inputs.Length][],  new double[outputs.Length][]);
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
    //Note: At no point should the input array be modified or returned.
    public double[] UnscaledInputs(double[] inputs)
    {
        inputs = Utilities.CopyNonObjectArray(inputs);
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
    //Note: At no point should the input array be modified or returned.
    public double[] ScaledInputs(double[] inputs)
    {
        inputs = Utilities.CopyNonObjectArray(inputs);
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
    //Note: At no point should the output array be modified or returned.
    public double[] UnscaledOutputs(double[] outputs)
    {
        outputs = Utilities.CopyNonObjectArray(outputs);
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
    //Note: At no point should the output array be modified or returned.
    public double[] ScaledOutputs(double[] outputs)
    {
        outputs = Utilities.CopyNonObjectArray(outputs);
        if (_outputScaling == null) return outputs;
        for (int i = 0; i < outputs.Length; i++)
            outputs[i] = (outputs[i] - _outputScaling[i].deshift) / _outputScaling[i].scale - _outputScaling[i].shift;
        return outputs;
    }
    
    /// <summary>
    /// Computes the predictions of this Network.
    /// </summary>
    /// <param name="inputs">The data to process.</param>
    /// <returns>The predictions of this Network for each piece of data.</returns>
    /// <exception cref="ArgumentException"></exception>
    //Note: At no point should the input array be modified or returned.
    //TODO: Implemented parallel processing for this method. Will need to check that no issues arise.
    //UPDATE: There was a fucking issue, and it took me 2 hours to realize this caused it.
    //Reverted this shit since I'm too pissed to fix it correctly.
    public double[][] Process(double[][] inputs)
    {
        double[][] outputs = new double[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i].Length != this[0].GetSize()) throw new ArgumentException($"Number of inputs does not match the size of the input Layer. (Sample #{i})");
            outputs[i] = ProcessSingle(inputs[i]);
        }
        return outputs;
    }
    
    /// <summary>
    /// Computes the predictions of this Network on a single data piece.
    /// </summary>
    /// <param name="inputs">The data to process.</param>
    /// <returns>The predictions of this Network.</returns>
    /// <exception cref="ArgumentException"></exception>
    //Note: At no point should the input array be modified or returned.
    internal double[] ProcessSingle(double[] inputs)
    {
        if (inputs.Length != this[0].GetSize()) throw new ArgumentException("Number of inputs does not match the size of the input Layer.");
        double[] current = UnscaledInputs(inputs);
        foreach (var layer in _networkLayers)
            current = layer.Process(current);
        return ScaledOutputs(current);
    }

    /// <summary>
    /// Computes the loss of this Network.
    /// </summary>
    /// <param name="data">The data.</param>
    /// <param name="lossType">The loss function.</param>
    /// <returns>The loss value of this Network on the specified data.</returns>
    /// <exception cref="ArgumentException"></exception>
    //Note: At no point should the input or output array be modified or returned.
    public double Loss(Dataset data, LossType lossType)
    {
        double totalError = 0;
        for (int i = 0; i < data.GetInputs().Length; i++)
        {
            if (data.GetOutputs()![i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this Network generates. (Sample #{i})");
            double[] scaledPredictions = ProcessSingle(data.GetInputs()[i]);
            double[] unscaledPredictions = UnscaledOutputs(scaledPredictions);
            double[] unscaledOutputs = UnscaledOutputs(data.GetOutputs()![i]);
            double entryLoss = 0;
            switch (lossType)
            {
                case LossType.MSE:
                    for (int j = 0; j < data.GetOutputs()![i].Length; j++)
                        entryLoss += LossFunction.MSE(unscaledPredictions[j], unscaledOutputs[j]);
                    entryLoss /= data.GetOutputs()![i].Length;
                    break;
                case LossType.BinaryCrossEntropy:
                    for (int j = 0; j < data.GetOutputs()![i].Length; j++)
                        entryLoss += LossFunction.BinaryCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                    break;
                case LossType.CategoricalCrossEntropy:
                    for (int j = 0; j < data.GetOutputs()![i].Length; j++)
                        entryLoss += LossFunction.CategoricalCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                    break;
            }
            totalError += entryLoss;
        }

        return totalError / data.GetInputs().Length;
    }
    
    /// <summary>
    /// Computes the losses of each data entry for this Network.
    /// </summary>
    /// <param name="data">The Dataset.</param>
    /// <param name="lossType">The loss function.</param>
    /// <returns>The losses for each data entry.</returns>
    /// <exception cref="ArgumentException"></exception>
    //Note: At no point should the input or output array be modified or returned.
    public double[] Losses(Dataset data, LossType lossType)
    {
        double[] losses = new double[data.GetInputs().Length];
        for (int i = 0; i < data.GetInputs().Length; i++)
        {
            if (data.GetOutputs()![i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this Network generates. (Sample #{i})");
            double[] scaledPredictions = ProcessSingle(data.GetInputs()[i]);
            double[] unscaledPredictions = UnscaledOutputs(scaledPredictions);
            double[] unscaledOutputs = UnscaledOutputs(data.GetOutputs()![i]);
            double entryLoss = 0;
            switch (lossType)
            {
                case LossType.MSE:
                    for (int j = 0; j < data.GetOutputs()![i].Length; j++)
                        entryLoss += LossFunction.MSE(unscaledPredictions[j], unscaledOutputs[j]);
                    entryLoss /= data.GetOutputs()![i].Length;
                    break;
                case LossType.BinaryCrossEntropy:
                    for (int j = 0; j < data.GetOutputs()![i].Length; j++)
                        entryLoss += LossFunction.BinaryCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                    break;
                case LossType.CategoricalCrossEntropy:
                    for (int j = 0; j < data.GetOutputs()![i].Length; j++)
                        entryLoss += LossFunction.CategoricalCrossEntropy(unscaledPredictions[j], unscaledOutputs[j]);
                    break;
            }
            losses[i] = entryLoss;
        }
        return losses;
    }
    
    /// <summary>
    /// The formatted string representing this Network.
    /// ⚠ This is a custom format, not JSON.
    /// </summary>
    /// <returns>A string representing this Network.</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append($"{_name ?? "null"};{_layerCount}\n");
        foreach (Layer layer in _networkLayers)
            sb.Append(layer);
        if (_inputScaling == null) sb.Append("null");
        else for (int i = 0; i < _inputScaling.Length; i++)
            sb.Append(_inputScaling[i].shift.ToString(CultureInfo.InvariantCulture) + "," + _inputScaling[i].scale.ToString(CultureInfo.InvariantCulture) + "," + _inputScaling[i].deshift.ToString(CultureInfo.InvariantCulture) + (i == _inputScaling.Length - 1 ? "" : ";"));
        sb.Append("\n");
        if (_outputScaling == null) sb.Append("null");
        else for (int i = 0; i < _outputScaling.Length; i++)
            sb.Append(_outputScaling[i].shift.ToString(CultureInfo.InvariantCulture) + "," + _outputScaling[i].scale.ToString(CultureInfo.InvariantCulture) + "," + _outputScaling[i].deshift.ToString(CultureInfo.InvariantCulture) + (i == _outputScaling.Length - 1 ? "" : ";"));
        return sb.ToString();
    }

    /// <summary>
    /// Randomizes all parameters of this Network in the specified range.
    /// </summary>
    /// <param name="min">The minimum of the range.</param>
    /// <param name ="max">The maximum of the range.</param>
    public void Randomize(double min, double max)
    {
        foreach (var layer in _networkLayers)
            foreach (var node in layer.GetNodes())
                for (int i = 0; i <= node.GetSize(); i++)
                    node[i] = Utilities.RandomDouble(min, max);
    }

    /// <summary>
    /// Clones this Network.
    /// </summary>
    /// <returns>A copy of this Network.</returns>
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
                ushort[]? parents = this[l, n].GetParents() == null ? null : new ushort[this[l, n].GetParentCount()];
                double[] weights = new double[this[l, n].GetSize()];
                if (parents != null) 
                    for (int p = 0; p < this[l, n].GetParentCount(); p++)
                        parents[p] = this[l, n].GetParents()![p];
                for (int w = 0; w < this[l, n].GetSize(); w++)
                    weights[w] = this[l, n].GetWeights()[w];
                network[l, n].SetParents(parents);
                network[l, n].SetWeights(weights);
                network[l, n].SetBias(this[l, n].GetBias());
            }
        }
        return network;
    }
}