namespace NeuralNetwork;

using System.Globalization;
using System;
using System.IO;
using System.Text;

public class Network
{
    // Fields
    private static Random _random = new Random();
    private static readonly object Lock = new object();

    private string? _name;

    private List<double>? _lossRecords;
    private ushort? _layerCount;
    private Layer[]? _networkLayers;
    private Layer? _inputLayer;
    private Layer? _outputLayer;

    private LossFunction? _lossFunction;
    private double? _baseLearningRate;

    //  Enums
    
    public enum LossFunction { CrossEntropy, MSE }

    // Constructors
    
    public Network(string? name)
    {
        _name = name ?? "NeuralNetwork";
    }
    

    //  Instantiation Methods

    public void InstantiateBasics(int hiddenLayers, LossFunction lossFunction, double? learningRate)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
        _lossFunction = lossFunction;
        _baseLearningRate = learningRate ?? 0.1;
        
    }
    

    // Setters
    
    public void SetLearningRate(double learningRate) => _baseLearningRate = learningRate;
    public void SetLoss(string loss) => _lossFunction = loss switch { "CrossEntropy" => LossFunction.CrossEntropy, _ => throw new ArgumentException("Invalid loss") };
    public void SetLayer(ushort i, Layer l) => _networkLayers![i] = l;
    public void SetNode(ushort layer, ushort node, ushort dimensions, Node.ActivationType activation, ushort[] parents) => _networkLayers![layer].GetNodes()[node] = new Node(layer, node, dimensions, activation, parents);
    public void SetName(string newName) => _name = newName;

    // Getters
    
    public double[] GetLossRecords() =>  _lossRecords!.ToArray();
    public string GetName() => _name!;
    public ushort GetLayerCount() => (ushort)_layerCount!;
    public Layer[] GetLayers() => _networkLayers!;
    public double GetLearningRate() => _baseLearningRate ?? 0.1;
    public LossFunction GetLossFunction() => _lossFunction ?? throw new NullReferenceException();

    // Indexers
    
    public Layer this[int layer]
    {
        get => _networkLayers![layer];
        set => _networkLayers![layer] = value;
    }

    public Node this[int layer, int node]
    {
        get => _networkLayers![layer].GetNodes()[node];
        set => _networkLayers![layer].SetNodes(node, value);
    }

    public double this[int layer, int node, int weightOrBias]
    {
        get => weightOrBias < 0
            ? _networkLayers![layer].GetNodes()[node].GetBias()
            : _networkLayers![layer].GetNodes()[node].GetWeights()[weightOrBias];
        set
        {
            if (weightOrBias < 0)
                _networkLayers![layer].GetNodes()[node].SetBias(value);
            else
                _networkLayers![layer].GetNodes()[node].GetWeights()[weightOrBias] = value;
        }
    }

    // Network Construction
    
    public void CreateInputLayer(ushort numberOfNodes, Node.ActivationType activation)
    {
        _inputLayer = new Layer(0, Layer.LayerType.Input);
        _inputLayer.Instantiate(numberOfNodes, 1, activation);
        _networkLayers![0] = _inputLayer;
    }

    public void CreateOutputLayer(ushort numberOfNodes, Node.ActivationType activation)
    {
        _outputLayer = new Layer((ushort)(_layerCount - 1)!, Layer.LayerType.Output);
        _outputLayer.Instantiate(numberOfNodes, _networkLayers![(int)(_layerCount - 2)!].GetSize(), activation);
        _networkLayers![(int)(_layerCount - 1)!] = _outputLayer;
    }

    public void CreateHiddenLayers(ushort numberOfNodes, Node.ActivationType activationType)
    {
        Layer afterInputLayer = new Layer(1, Layer.LayerType.Hidden);
        afterInputLayer.Instantiate(numberOfNodes, _networkLayers![0].GetSize(), activationType);
        _networkLayers![1] = afterInputLayer;
        for (int i = 2; i < _layerCount - 1; i++)
        {
            Layer hiddenLayer = new Layer((ushort)i, Layer.LayerType.Hidden);
            hiddenLayer.Instantiate(numberOfNodes, numberOfNodes, activationType);
            _networkLayers![i] = hiddenLayer;
        }
    }
    
    public void CreateLayer(int identifier, ushort numberOfNodes, Node.ActivationType activationType)
    {
        Layer layer = new Layer((ushort)(_layerCount - 1)!, identifier == 0 ? Layer.LayerType.Input : identifier == _layerCount - 1 ? Layer.LayerType.Output : Layer.LayerType.Hidden);
        layer.Instantiate(numberOfNodes, _networkLayers![identifier - 1].GetSize(), activationType);
        _networkLayers![(int)(_layerCount - 1)!] = layer;
    }

    // Network Processing
    
    public double[][] Process(double[][] features)
    {
        List<double[]> results = new List<double[]>();
        for (int i = 0; i < features.Length; i++)
        {
            double[] input = new double[features[i].Length];
            for (int j = 0; j < features[i].Length; j++)
                input[j] = features[i][j];
            foreach (var layer in _networkLayers!)
                input = layer.Process(input);
            results.Add(input);
        }
        return results.ToArray();
    }

    private double[] ProcessSingle(double[] features)
    {
        double[] input = new double[features.Length];
        for (int i = 0; i < features.Length; i++)
            input[i] = features[i];
        foreach (var layer in _networkLayers!)
            input = layer.Process(input);
        return input;
    }
    
    // Evaluation
    public double Loss(double[][] features, double[][] expectedOutputs)
    {
        double totalError = 0;
        for (int i = 0; i < features.Length; i++)
        {
            double currError = 0;
            double[] predicted = ProcessSingle(features[i]);
            for (int j = 0; j < expectedOutputs[i].Length; j++)
                currError += Math.Abs(expectedOutputs[i][j] - predicted[j]);
            currError /= expectedOutputs[i].Length;
            totalError += currError;
        }
        totalError /= features.Length;
        return totalError;
    }

    public void LossWithPrint(double[][] features, double[][] expectedOutputs)
    {
        double totalError = 0;
        for (int i = 0; i < features.Length; i++)
        {
            double currError = 0;
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < features[i].Length; j++)
                sb.Append(features[i][j] + ",");
            string input = sb.ToString();
            double[] predicted = ProcessSingle(features[i]);
            Console.WriteLine($"Input: {input} Predicted: {string.Join(", ", predicted)} Expected: {string.Join(", ", expectedOutputs[i])}");
            for (int j = 0; j < expectedOutputs[i].Length; j++)
                currError += Math.Abs(expectedOutputs[i][j] - predicted[j]);
            currError /= expectedOutputs[i].Length;
            totalError += currError;
        }
        totalError /= features.Length;
        Console.WriteLine($"Loss: {100 *  totalError}%");
    }
    
    // Optimization

    public void Optimize(double[][] features, double[][] expectedOutputs, uint totalEpochs, bool countLosses)
    {
        if (countLosses) _lossRecords = new List<double>();
        double previousLoss = 1;
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            int sampleCount = expectedOutputs.Length;
            int batchSize = 64;
            int batches = (sampleCount + batchSize - 1) / batchSize;
            for (int batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                double[][][] weightGradientsForBatch = InstantiateWeightArray();
                double[][] biasGradientsForBatch = InstantiateBiasArray();
                for (int sampleIndex = batchIndex * batchSize; sampleIndex < sampleCount && sampleIndex < batchIndex * batchSize + batchSize; sampleIndex++)
                {
                    double[][][] weightGradientsPerLayer = InstantiateWeightArray();
                    double[][] biasGradientsPerLayer = InstantiateBiasArray();
                    NodeActivationRecord[][] forwardPassRecords = new NodeActivationRecord[(int)_layerCount!][];
                    double[] layerInputs = features[sampleIndex];
                    double[] predictions = ForwardPass(ref forwardPassRecords, layerInputs);
                    double[] nextLayerDeltas = OutputLayerGradients(predictions, expectedOutputs, sampleIndex);
                    BackPropagation(nextLayerDeltas, forwardPassRecords, weightGradientsPerLayer, biasGradientsPerLayer);
                    Accumulate(ref weightGradientsForBatch, ref biasGradientsForBatch, weightGradientsPerLayer, biasGradientsPerLayer);
                }
                UpdateGradients(weightGradientsForBatch, biasGradientsForBatch, batchSize);
            }
            if (epoch % 100 == 0)
            {
                _baseLearningRate *= 0.995;
                double cur = Loss(features, expectedOutputs);
                if (previousLoss - cur < 1e-6) break;
                previousLoss = cur;
            }
        }
    }

    private void BackPropagation(double[] nextLayerDeltas, NodeActivationRecord[][] forwardPassRecords, double[][][] weightGradientsPerLayer, double[][] biasGradientsPerLayer)
    {

        for (int layerIndex = _networkLayers.Count() - 1; layerIndex >= 0; layerIndex--)
        {
            Layer currentLayer = _networkLayers[layerIndex];
            NodeActivationRecord[] layerActivationRecords = forwardPassRecords[layerIndex];
            double[] currentLayerDeltas = new double[currentLayer.GetSize()];

            for (int nodeIndex = 0; nodeIndex < currentLayer.GetSize(); nodeIndex++)
            {
                Node currentNode = currentLayer.GetNodes()[nodeIndex];
                NodeActivationRecord activationRecord = layerActivationRecords[nodeIndex];

                // Calculate delta for current node
                double delta;
                if (layerIndex == _networkLayers.Count() - 1) // Output layer
                    delta = nextLayerDeltas[nodeIndex];
                else // Hidden layers
                {
                    delta = 0;
                    Layer downstreamLayer = _networkLayers[layerIndex + 1];
                    foreach (Node downstreamNode in downstreamLayer.GetNodes())
                        for (int parentIndex = 0; parentIndex < downstreamNode.GetParents().Length; parentIndex++)
                            if (downstreamNode.GetParents()[parentIndex] == nodeIndex)
                                delta += downstreamNode.GetWeights()[parentIndex] * nextLayerDeltas[Array.IndexOf(downstreamLayer.GetNodes(), downstreamNode)];
                    delta *= activationRecord.ActivationDerivative;
                }
                currentLayerDeltas[nodeIndex] = delta;

                for (int weightIndex = 0; weightIndex < currentNode.GetDimensions(); weightIndex++)
                    weightGradientsPerLayer[layerIndex][nodeIndex][weightIndex] +=
                        delta * activationRecord.InputValues[weightIndex];
                biasGradientsPerLayer[layerIndex][nodeIndex] += delta;
            }
            nextLayerDeltas = currentLayerDeltas;
        }
    }

    private double[] OutputLayerGradients(double[] layerInputs, double[][] expectedOutputs, int sampleIndex)
    {
        double[] nextLayerDeltas = new double[layerInputs.Length];
        Layer outputLayer = _networkLayers!.Last();

        for (int i = 0; i < outputLayer.GetSize(); i++)
        {
            double output = layerInputs[i];
            Node node = outputLayer.GetNodes()[i];

            switch (_lossFunction)
            {
                case LossFunction.CrossEntropy:
                    if (node.GetActivation() == Node.ActivationType.Sigmoid || node.GetActivation() == Node.ActivationType.Softmax) nextLayerDeltas[i] = output - expectedOutputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
                case LossFunction.MSE:
                    if (node.GetActivation() == Node.ActivationType.Linear) nextLayerDeltas[i] = output - expectedOutputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
            }
        }
        return nextLayerDeltas;
    }

    private double[] ForwardPass(ref NodeActivationRecord[][] forwardPassRecords, double[] layerInputs)
    {
        for (int l = 0; l < _layerCount; l++)
        {
            Layer layer = this[l];
            Node[] nodes = layer.GetNodes();
            NodeActivationRecord[] layerRecord = new NodeActivationRecord[layer.GetSize()];
            double[] layerOutputs = new double[layer.GetSize()];
            double[] weightedSums = layer.WeightedSums(layerInputs);
            double[]? softmaxOutputs = layer[0].GetActivation() == Node.ActivationType.Softmax ? layer.SoftmaxOutputs(weightedSums) : null;
            for (int n = 0; n < layer.GetSize(); n++)
            {
                            
                Node node = nodes[n];
                layerRecord[n] = NodeRecord(node, weightedSums, softmaxOutputs, n, layer, layerInputs);
                layerOutputs[n] = layerRecord[n].ActivationOutput;
            }

            forwardPassRecords[l] = layerRecord;
            layerInputs = layerOutputs;
        }
        return layerInputs;
    }

    private NodeActivationRecord NodeRecord(Node node, double[] weightedSums, double[]? softmaxOutputs, int n, Layer layer, double[] inputs)
    {
        double activationOutput = 0;
        double activationDerivative = 0;
        switch (node.GetActivation())
        {
            case Node.ActivationType.Softmax:
                activationOutput = softmaxOutputs![n];
                activationDerivative = 1;
                break;
            case Node.ActivationType.Sigmoid:
                activationOutput = Functions.Sigmoid(weightedSums[n]);
                activationDerivative = activationOutput * (1 - activationOutput);
                break;
            case Node.ActivationType.Linear:
                activationOutput = weightedSums[n];
                activationDerivative = 1;
                break;
            case Node.ActivationType.RElu:
                activationOutput = Math.Max(0, weightedSums[n]);
                activationDerivative = weightedSums[n] > 0 ? 1 : 0;
                break;
            case Node.ActivationType.LeakyRElu:
                activationOutput = Math.Max(0, weightedSums[n]);
                activationDerivative = weightedSums[n] > 0 ? 1 : 0.01;
                break;
            case Node.ActivationType.Tanh:
                activationOutput = Math.Tanh(weightedSums[n]);
                activationDerivative = 1 - activationOutput * activationOutput;
                break;
            // TODO: Other derivatives
        }
        return new NodeActivationRecord { InputValues = layer.NodeInputs(inputs, n), WeightedSum = weightedSums[n], ActivationOutput = activationOutput, ActivationDerivative = activationDerivative };
    }

    private double[][][] InstantiateWeightArray()
    {
        double[][][] arr = new double[(int)_layerCount!][][];
        for (int l = 0; l < _layerCount; l++)
        {
            arr[l] = new double[this[l].GetSize()][];
            for (int n = 0; n < this[l].GetSize(); n++)
            {
                arr[l][n] = new double[this[l, n].GetDimensions()];
            }
        }
        return arr;
    }

    private double[][] InstantiateBiasArray()
    {
        double[][] arr = new double[(int)_layerCount!][];
        for (int l = 0; l < _layerCount; l++)
            arr[l] = new double[this[l].GetSize()];
        return arr;
    }

    private void Accumulate(ref double[][][] weightAccumulator, ref double[][] biasAccumulator, double[][][] weights, double[][] biases)
    {
        for (int l = 0; l < _networkLayers.Length; l++)
        {
            for (int n = 0; n < _networkLayers[l].GetSize(); n++)
            {
                biasAccumulator[l][n] += biases[l][n];
                for (int w = 0; w < weights[l][n].Length; w++)
                {
                    weightAccumulator[l][n][w] += weights[l][n][w];
                }
            }
        }
    }

    private void UpdateGradients(double[][][] totalWeightGradients, double[][] totalBiasGradients, int sampleCount)
    {
        for (int layerIndex = 0; layerIndex < _networkLayers.Count(); layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < this[layerIndex].GetSize(); nodeIndex++)
            {
                this[layerIndex, nodeIndex, -1] -= (double)_baseLearningRate! * totalBiasGradients[layerIndex][nodeIndex] / sampleCount;
                for (int weightIndex = 0; weightIndex < this[layerIndex, nodeIndex].GetDimensions(); weightIndex++)
                    this[layerIndex, nodeIndex, weightIndex] -= (double)_baseLearningRate * totalWeightGradients[layerIndex][nodeIndex][weightIndex] / sampleCount;
            }
        }
    }
  

    private double GetDerivative(Node node, double output)
    {
        switch (node.GetActivation())
        {
            case Node.ActivationType.Sigmoid:
                return output * (1 - output);
            case Node.ActivationType.LeakyRElu:
                return output > 0 ? 1 : 0.01;
            case Node.ActivationType.RElu:
                return output > 0 ? 1 : 0;
            case Node.ActivationType.Tanh:
                return 1 - output * output;
            case Node.ActivationType.Linear:
                return 1;
            default:
                return 1;
        }
    }
    

    private class NodeActivationRecord
    {
        public required double[] InputValues { get; init; } // Inputs received from previous layer
        public double WeightedSum { get; set; } // z = sum(weights * inputs) + bias
        public double ActivationOutput { get; set; } // a = activation(z)
        public double ActivationDerivative { get; init; } // da/dz derivative at z
    }

    // Saving & Loading
    
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append($"{_name};{_layerCount}\n");
        foreach (Layer layer in _networkLayers!)
            sb.Append(layer);
        sb.Append($"{_lossFunction};{_baseLearningRate?.ToString(CultureInfo.InvariantCulture)}");
        return sb.ToString();
    }

    public void SaveToFile(string filePath) => File.WriteAllText(filePath, ToString());

    private void LoadFromFile(string filePath)
    {
        string[] lines = File.ReadAllLines(filePath);
        if (lines.Length == 0)
            throw new InvalidDataException("File is empty.");

        // Parse the first line (network metadata)
        string[] header = lines[0].Split(';');
        if (header.Length < 3)
            throw new InvalidDataException("Invalid header format.");

        _name = header[0];
        _layerCount = ushort.Parse(header[1]);
        _networkLayers = new Layer[_layerCount.Value];

        int currentLine = 1; // Start reading layers after the header

        // Parse each layer
        for (int layerIdx = 0; layerIdx < _layerCount; layerIdx++)
        {
            if (currentLine >= lines.Length)
                throw new InvalidDataException("Unexpected end of file while reading layers.");

            string[] layerHeader = lines[currentLine].Split(';');
            if (layerHeader.Length < 2)
                throw new InvalidDataException($"Invalid layer header at line {currentLine}.");

            Layer.LayerType layerType = Enum.Parse<Layer.LayerType>(layerHeader[0]);
            ushort layerSize = ushort.Parse(layerHeader[1]);
            currentLine++;

            Layer layer = new Layer((ushort)layerIdx, layerType);
            layer.InstantiateCustom(layerSize);
            _networkLayers[layerIdx] = layer;

            // Parse nodes in this layer
            for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
            {
                if (currentLine >= lines.Length)
                    throw new InvalidDataException("Unexpected end of file while reading nodes.");

                string[] nodeData = lines[currentLine].Split(';');
                if (nodeData.Length < 5)
                    throw new InvalidDataException($"Invalid node data at line {currentLine}.");

                ushort dimensions = ushort.Parse(nodeData[0]);
                double[] weights = nodeData[1].Split(',')
                    .Select(w => double.Parse(w, CultureInfo.InvariantCulture))
                    .ToArray();
                double bias = double.Parse(nodeData[2], CultureInfo.InvariantCulture);
                Node.ActivationType activation = Enum.Parse<Node.ActivationType>(nodeData[3]);
                ushort[] parents = nodeData[4].Split(',')
                    .Select(p => ushort.Parse(p))
                    .ToArray();

                Node node = new Node(
                    (ushort)nodeIndex,
                    (ushort)layerIdx,
                    dimensions,
                    weights,
                    bias,
                    activation,
                    parents
                );

                layer.SetNodes(nodeIndex, node);
                currentLine++;
            }
        }

        // Parse loss function and learning rate (last line)
        if (currentLine >= lines.Length)
            throw new InvalidDataException("Missing loss function and learning rate.");

        string[] footer = lines[currentLine].Split(';');
        if (footer.Length < 2)
            throw new InvalidDataException("Invalid footer format.");

        _lossFunction = Enum.Parse<LossFunction>(footer[0]);
        _baseLearningRate = double.Parse(footer[1], CultureInfo.InvariantCulture);

        // Set input/output layers
        _inputLayer = _networkLayers[0];
        _outputLayer = _networkLayers[_networkLayers.Length - 1];
    }

    // Randomization / Testing

    public void Randomize(double range)
    {
        foreach (var layer in _networkLayers!)
        {
            foreach (var node in layer.GetNodes())
            {
                if (range == 0)
                {
                    switch (node.GetActivation())
                    {
                        case Node.ActivationType.RElu:
                            range = Math.Sqrt(2.0 / node.GetDimensions());
                            break;
                        case Node.ActivationType.Sigmoid:
                            range = Math.Sqrt(6.0 / (node.GetDimensions() + 1));
                            break;
                        default:
                            range = 1;
                            break;
                    }
                }
                for (int i = 0; i < node.GetDimensions(); i++)
                    node.GetWeights()[i] = Network.RandomDouble(-range, range);
                node.SetBias(0);
            }
        }
    }

    private (double[], double)[][] StoreSettings()
    {
        return _networkLayers!
            .Select(layer => layer.GetNodes()
                .Select(node => (
                    node.GetWeights().ToArray(),
                    node.GetBias()
                )).ToArray())
            .ToArray();
    }

    private void ExtractSettings((double[], double)[][] settings)
    {
        for (int l = 0; l < _networkLayers!.Length; l++)
        {
            for (int n = 0; n < _networkLayers[l].GetSize(); n++)
            {
                _networkLayers[l].GetNodes()[n].SetBias(settings[l][n].Item2);
                for (int w = 0; w < _networkLayers[l].GetNodes()[n].GetDimensions(); w++)
                    _networkLayers[l].GetNodes()[n].GetWeights()[w] = settings[l][n].Item1[w];
            }
        }
    }

    // Utilities

    public static double RandomDouble(double minValue, double maxValue)
    {
        lock (Lock)
        {
            return _random.NextDouble() * (maxValue - minValue) + minValue;
        }
    }

    public static void NormalizeData(double[][] data)
    {
        if (data == null || data.Length == 0) return;
        int rows = data.Length;
        int cols = data[0].Length;
        double[] min = new double[cols];
        double[] max = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            min[j] = double.MaxValue;
            max[j] = double.MinValue;
            for (int i = 0; i < rows; i++)
            {
                if (data[i][j] < min[j]) min[j] = data[i][j];
                if (data[i][j] > max[j]) max[j] = data[i][j];
            }
        }
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double range = max[j] - min[j];
                if (range == 0)
                    data[i][j] = 0;
                else
                    data[i][j] = (data[i][j] - min[j]) / range;
            }
        }
    }
}
