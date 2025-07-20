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
    /// ⚠ This method returns a reference to the actual Layer this Network uses, modifying it will modify the that of Network too.
    /// </summary>
    /// <param name="layerIndex">The index of the Layer.</param>
    public Layer this[int layerIndex]
    {
        get => _networkLayers[layerIndex];
        set => _networkLayers[layerIndex] = value;
    }
    
    /// <summary>
    /// Indexer for this Network's Nodes.
    /// ⚠ This method returns a reference to the actual Node this Network uses, modifying it will modify the that of Network too.
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
    public string GetName()
    {
        if (_name == null) throw new Exception("Network doesn't have a name.");
        return _name;
    }
    
    /// <summary>
    /// Sets the name of this Network.
    /// </summary>
    /// <param name="name">The new name of this Network</param>
    public void SetName(string name) => _name = name;
    
    /// <summary>
    /// Sets the input scaling of this Network.
    /// </summary>
    /// <param name="scales">The new input scales.</param>
    public void SetInputScaling((double, double, double)[]? scales) => _inputScaling = scales == null ? null : NetworkUtilities.CopyNonObjectArray(scales);
    
    /// <summary>
    /// Sets the output scaling of this Network.
    /// </summary>
    /// <param name="scales">The new output scales.</param>
    public void SetOutputScaling((double, double, double)[]? scales) => _outputScaling = scales == null ? null : NetworkUtilities.CopyNonObjectArray(scales);
    
    /// <summary>
    /// Fetches the Layer count of this Network.
    /// </summary>
    /// <returns>This Network's Layer count.</returns>
    public ushort GetLayerCount() => _layerCount;
    
    /// <summary>
    /// Fetches the Layer array of this Network.
    /// ⚠ This method returns a reference to the actual array this Network uses, modifying it will modify the that of Network too.
    /// </summary>
    /// <returns>This Network's Layer array./// </returns>
    public Layer[] GetLayers() => _networkLayers;

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
    public double[] UnscaledInputs(double[] inputs)
    {
        inputs = NetworkUtilities.CopyNonObjectArray(inputs);
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
        inputs = NetworkUtilities.CopyNonObjectArray(inputs);
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
        outputs = NetworkUtilities.CopyNonObjectArray(outputs);
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
        outputs = NetworkUtilities.CopyNonObjectArray(outputs);
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
    //TODO: Implemented parallel processing for this method. Will need to check that no issues arise.
    //UPDATE: There was a fucking issue, and it took me 2 hours to realize this caused it.
    //Reverted this shit since I'm too pissed to fix it correctly
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
    /// <param name="inputs">The inputs of data.</param>
    /// <param name="outputs">The outputs of data.</param>
    /// <param name="lossType">The loss function.</param>
    /// <returns>The loss value of this Network on the specified data.</returns>
    /// <exception cref="ArgumentException"></exception>
    public double Loss(double[][] inputs, double[][] outputs, LossType lossType)
    {
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this Network generates. (Sample #{i})");
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
    /// Computes the loss of this Network with insights on the predictions.
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
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this Network generates. (Sample #{i})");
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
            sb.Append($"#{i} Input: ");
            for (int k = 0; k < inputs[i].Length; k++)
                sb.Append(inputs[i][k].ToString("F6", CultureInfo.InvariantCulture) + ";");
            sb.Append(" Predicted: ");
            for (int k = 0; k < scaledPredictions.Length; k++)
                sb.Append(scaledPredictions[k].ToString("F6", CultureInfo.InvariantCulture) + ";");
            sb.Append(" Expected: ");
            for (int k = 0; k < scaledOutputs.Length; k++)
                sb.Append(scaledOutputs[k].ToString("F6", CultureInfo.InvariantCulture) + ";");
            sb.AppendLine($" Loss: {currError.ToString("F6", CultureInfo.InvariantCulture)} Total: {totalError.ToString("F6", CultureInfo.InvariantCulture)}");
            
        }
        totalError /= inputs.Length;
        sb.AppendLine($"Loss: {totalError}");
        return sb.ToString();
    }
    
    /// <summary>
    /// The formatted string representing this Network.
    /// ⚠ This is a custom format, not JSON.
    /// </summary>
    /// <returns>A string representing this Network.</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append($"{_name};{_layerCount}\n");
        foreach (Layer layer in _networkLayers)
            sb.Append(layer);
        sb.AppendLine("#SCALES");
        if (_inputScaling == null) sb.AppendLine("#");
        else for (int i = 0; i < _inputScaling.Length; i++)
            sb.AppendLine(_inputScaling[i].shift + "," + _inputScaling[i].scale + "," + _inputScaling[i].deshift + (i == _inputScaling.Length - 1 ? "" : ";"));
        if (_outputScaling == null) sb.AppendLine("#");
        else for (int i = 0; i < _outputScaling.Length; i++)
            sb.Append(_outputScaling[i].shift + "," + _outputScaling[i].scale + "," + _outputScaling[i].deshift + (i == _outputScaling.Length - 1 ? "" : ";"));
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
                    node[i] = NetworkUtilities.NextDouble(min, max);
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