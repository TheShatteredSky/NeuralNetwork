namespace NeuralNetwork.Core;

public class Network
{
    private string? _name;
    private ushort _layerCount;
    private Layer[] _networkLayers;
    private (double shift, double scale, double deshift)[] _inputScaling;
    private (double shift, double scale, double deshift)[] _outputScaling;

    public Network()
    {
        _name = null;
        _layerCount = 0;
        _networkLayers = [];
        _inputScaling = [];
        _outputScaling = [];
    }

    public Network(string name)
    {
        _name = name;
        _layerCount = 0;
        _networkLayers = [];
        _inputScaling = [];
        _outputScaling = [];
    }

    public void Instantiate(int hiddenLayers)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
    }

    public void Instantiate(ushort inputSize, ActivationType inputActivation, int hiddenLayers, ushort hiddenSize, ActivationType hiddenActivation, ushort outputSize, ActivationType outputActivation)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
        CreateInputLayer(inputSize, inputActivation);
        CreateHiddenLayers(hiddenSize, hiddenActivation);
        CreateOutputLayer(outputSize, outputActivation);
    }
    
    public void SetName(string newName) => _name = newName;
    
    public void SetInputScaling((double, double, double)[] scales) => _inputScaling = scales;
    
    public void SetOutputScaling((double, double, double)[] scales) => _outputScaling = scales;
    
    public string GetName() => _name!;
    public ushort GetLayerCount() => _layerCount;
    public Layer[] GetLayers() => _networkLayers;

    public Layer this[int layer]
    {
        get => _networkLayers[layer];
        set => _networkLayers[layer] = value;
    }

    public Node this[int layer, int node]
    {
        get => this[layer][node];
        set => this[layer][node] = value;
    }

    public double this[int layer, int node, int param]
    {
        get => this[layer, node][param];
        set => this[layer, node][param] = value;
    }

    public void CreateInputLayer(ushort numberOfNodes, ActivationType activation)
    {
        _inputScaling = new (double shift, double scale, double deshift)[numberOfNodes];
        for (int i = 0; i < numberOfNodes; i++)
            _inputScaling[i] = (0, 1, 0);
        if (activation == ActivationType.Softmax) throw new ArgumentException("Input layers cannot have a Softmax activation.");
        Layer input = new Layer(0, Layer.LayerType.Input);
        input.Instantiate(numberOfNodes, 1, activation);
        _networkLayers[0] = input;
    }
    
    public void CreateHiddenLayers(ushort numberOfNodes, ActivationType activation)
    {
        if (activation == ActivationType.Softmax) throw new ArgumentException("Hidden layers cannot have a Softmax activation.");
        Layer afterInputLayer = new Layer(1, Layer.LayerType.Hidden);
        afterInputLayer.Instantiate(numberOfNodes, _networkLayers[0].GetSize(), activation);
        _networkLayers[1] = afterInputLayer;
        for (int i = 2; i < _layerCount - 1; i++)
        {
            Layer hiddenLayer = new Layer((ushort)i, Layer.LayerType.Hidden);
            hiddenLayer.Instantiate(numberOfNodes, numberOfNodes, activation);
            _networkLayers[i] = hiddenLayer;
        }
    }

    public void CreateOutputLayer(ushort numberOfNodes, ActivationType activation)
    {
        _outputScaling = new (double shift, double scale, double deshift)[numberOfNodes];
        for (int i = 0; i < numberOfNodes; i++)
            _outputScaling[i] = (0, 1, 0);
        Layer output = new Layer((ushort)(_layerCount - 1)!, Layer.LayerType.Output);
        output.Instantiate(numberOfNodes, _networkLayers[_layerCount - 2].GetSize(), activation);
        _networkLayers[_layerCount - 1] = output;
    }

    public (double[][] scaledInputs, double[][] scaledOutputs) ScaledData(double[][] inputs, double[][] outputs)
    {
        (double[][] scaledInputs, double[][] scaledOutputs) scaledData = new();
        scaledData.scaledInputs = new double[inputs.Length][];
        scaledData.scaledOutputs = new double[outputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            scaledData.scaledInputs[i] = ScaledInputs(inputs[i]);
        for (int i = 0; i < outputs.Length; i++)
            scaledData.scaledOutputs[i] = ScaledOutputs(outputs[i]);
        return scaledData;
    }

    public double[] ScaledInputs(double[] inputs)
    {
        double[] scaledInputs = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                scaledInputs[i] = (inputs[i] + _inputScaling[i].shift)  * _inputScaling[i].scale + _inputScaling[i].deshift;
        return scaledInputs;
    }

    public double[] ScaledOutputs(double[] outputs)
    {
        double[] scaledOutputs = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
                scaledOutputs[i] = (outputs[i] + _outputScaling[i].shift)  * _outputScaling[i].scale + _outputScaling[i].deshift;
        return scaledOutputs;
    }
    
    public double[] UnscaledOutputs(double[] outputs)
    {
        double[] unscaledOutputs = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
                unscaledOutputs[i] = (outputs[i] - _outputScaling[i].deshift) / _outputScaling[i].scale - _outputScaling[i].shift;
        return unscaledOutputs;
    }

    public double[][] Process(double[][] inputs)
    { 
        double[][] outputs = new double[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i].Length != this[0].GetSize()) throw new ArgumentException($"Number of inputs does not match the size of the input layer. (Sample #{i})");
            outputs[i] = ProcessSingle(inputs[i]);
        }
        return outputs;
    }

    private double[] ProcessSingle(double[] inputs)
    {
        if (inputs.Length != this[0].GetSize()) throw new ArgumentException("Number of inputs does not match the size of the input layer.");
        double[] current = ScaledInputs(inputs);
        foreach (var layer in _networkLayers)
            current = layer.Process(current);
        return UnscaledOutputs(current);
    }

    internal double[][] ProcessPreScaled(double[][] preScaledInputs)
    {
        double[][] outputs = new double[preScaledInputs.Length][];
        for (int i = 0; i < preScaledInputs.Length; i++)
        {
            if (preScaledInputs[i].Length != this[0].GetSize()) throw new ArgumentException($"Number of inputs does not match the size of the input layer. (Sample #{i})");
            outputs[i] = ProcessSinglePreScaled(preScaledInputs[i]);
        }
        return outputs;
    }
    
    private double[] ProcessSinglePreScaled(double[] preScaledInputs)
    {
        if (preScaledInputs.Length != this[0].GetSize()) throw new ArgumentException("Number of inputs does not match the size of the input layer.");
        double[] current = preScaledInputs;
        foreach (var layer in _networkLayers)
            current = layer.Process(current);
        return current;
    }

    public double Loss(double[][] inputs, double[][] outputs, LossType lossType)
    {
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this network generates. (Sample #{i})");
            double[] predictions = ProcessSingle(inputs[i]);
            double[] scaledOutputs = ScaledOutputs(outputs[i]);
            double sampleLoss = 0;
            switch (lossType)
            {
                case LossType.MSE:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.MSE(predictions[j], scaledOutputs[j]);
                    sampleLoss /= outputs[i].Length;
                    break;
                case LossType.BinaryCrossEntropy:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.BinaryCrossEntropy(predictions[j],  scaledOutputs[j]);
                    break;
                case LossType.CategoricalCrossEntropy:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.CategoricalCrossEntropy(predictions[j],  scaledOutputs[j]);
                    break;
            }
            totalError += sampleLoss;
        }

        return totalError / inputs.Length;
    }
    
    public string LossString(double[][] inputs, double[][] outputs, LossType lossFunction)
    {
        StringBuilder sb = new StringBuilder();
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this network generates. (Sample #{i})");
            double currError = 0;
            double[] predictions = ProcessSingle(inputs[i]);
            double[] scaledOutputs = ScaledOutputs(outputs[i]);
           
            for (int j = 0; j < outputs[i].Length; j++)
            {
                switch (lossFunction)
                {
                    case LossType.MSE:
                        currError += LossFunction.MSE(predictions[j], scaledOutputs[j]);
                        currError /= outputs[i].Length;
                        break;
                    case LossType.BinaryCrossEntropy:
                        currError += LossFunction.BinaryCrossEntropy(predictions[j], scaledOutputs[j]);
                        break;
                    case LossType.CategoricalCrossEntropy:
                        currError += LossFunction.CategoricalCrossEntropy(predictions[j], scaledOutputs[j]);
                        break;
                }
            }
            totalError += currError;
            sb.AppendLine($"#{i} Input: {string.Join(", ", inputs[i])} Predicted: {string.Join(", ", predictions)} Expected: {string.Join(", ", scaledOutputs)} Loss: {currError} Total: {totalError}");
        }
        totalError /= inputs.Length;
        sb.AppendLine($"Loss: {totalError}");
        return sb.ToString();
    }
    
    public double AbsoluteLoss(double[][] inputs, double[][] outputs)
    {
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this network generates. (Sample #{i})");
            double currentError = 0;
            double[] predictions = ProcessSingle(inputs[i]);
            double[] scaledOutputs = ScaledOutputs(outputs[i]);
            for (int j = 0; j < outputs[i].Length; j++)
                currentError += Math.Abs(scaledOutputs[j] - predictions[j]);
            currentError /= outputs[i].Length;
            totalError += currentError;
        }
        totalError /= inputs.Length;
        return totalError;
    }

    public string AbsoluteLossString(double[][] inputs, double[][] outputs)
    {
        StringBuilder sb = new StringBuilder();
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            if (outputs[i].Length != this[_layerCount - 1].GetSize()) throw new ArgumentException($"Number of expected outputs does not match the number of outputs this network generates. (Sample #{i})");
            double currError = 0;
            double[] predictions = ProcessSingle(inputs[i]);
            double[] scaledOutputs = ScaledOutputs(outputs[i]);
            for (int j = 0; j < outputs[i].Length; j++)
                currError += Math.Abs(scaledOutputs[j] - predictions[j]);
            currError /= outputs[i].Length;
            totalError += currError;
            sb.AppendLine($"#{i} Input: {string.Join(", ", inputs[i])} Predicted: {string.Join(", ", predictions)} Expected: {string.Join(", ", scaledOutputs)} Loss: {currError} Total: {totalError}");
        }
        totalError /= inputs.Length;
        sb.AppendLine($"Loss: {100 * totalError}%");
        return sb.ToString();
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append($"{_name};{_layerCount}\n");
        foreach (Layer layer in _networkLayers)
            sb.Append(layer);
        sb.AppendLine();
        for (int i = 0; i < _inputScaling.Length; i++)
            sb.Append(_inputScaling[i].shift + "," + _inputScaling[i].scale + "," + _inputScaling[i].deshift + ";");
        sb.AppendLine();
        for (int i = 0; i < _outputScaling.Length; i++)
            sb.Append(_outputScaling[i].shift + "," + _outputScaling[i].scale + "," + _outputScaling[i].deshift + ";");
        return sb.ToString();
    }

    public void Randomize(double range)
    {
        foreach (var layer in _networkLayers)
        {
            foreach (var node in layer.GetNodes())
            {
                for (int i = 0; i < node.GetDimensions(); i++)
                    node.GetWeights()[i] = NetworkUtilities.NextDouble(-range, range);
                node.SetBias(0);
            }
        }
    }
}