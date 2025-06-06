namespace NeuralNetwork;

using System.Globalization;
using System;
using System.IO;
using System.Text;

public class NeuralNetwork
{
    // Fields
    private static Random _random = new Random();
    private static readonly object Lock = new object();

    private string? _name;
    private NetworkType _type;

    private ushort? _layerCount;
    private Layer[]? _networkLayers;
    private Layer? _inputLayer;
    private Layer? _outputLayer;

    private LossFunction? _lossFunction;
    private double? _learningRate;

    //  Enums
    
    public enum NetworkType { Custom, BinaryClassification, GeneralClassification }
    public enum LossFunction { CrossEntropy, MSE }

    // Constructors
    
    public NeuralNetwork(string? name, NetworkType type)
    {
        _name = name ?? "NeuralNetwork";
        _type = type;
    }

    public NeuralNetwork(string filePath)
    {
        //LoadFromFile(filePath);
    }

    //  Instantiation Methods

    public void InstantiateBinaryClassification(ushort hiddenLayers, ushort hiddenLayersSize, ushort inputLayerSize)
    {
         /* ... */
    }

    public void InstantiateGeneralClassification(ushort hiddenLayers, ushort hiddenLayersSize, ushort inputLayerSize, ushort outputLayerSize)
    {
         /* ... */
    }

    // Setters
    
    public void SetLearningRate(double learningRate) => _learningRate = learningRate;
    public void SetLoss(string loss) => _lossFunction = loss switch { "CrossEntropy" => LossFunction.CrossEntropy, _ => throw new ArgumentException("Invalid loss") };
    public void SetLayer(ushort i, Layer l) => _networkLayers![i] = l;
    public void SetNeuron(ushort layer, ushort neuron, ushort dimensions, Neuron.ActivationType activation, ushort[] parents, bool hasParents) => _networkLayers![layer].GetNeurons()[neuron] = new Neuron(layer, neuron, dimensions, activation, parents, hasParents);

    // Getters
    
    public string GetName() => _name!;
    public ushort GetLayerCount() => (ushort)_layerCount!;
    public Layer[] GetLayers() => _networkLayers!;
    public double GetLearningRate() => _learningRate ?? 0.1;
    public LossFunction GetLossFunction() => _lossFunction ?? throw new NullReferenceException();

    // Indexers
    
    public Layer this[int layer]
    {
        get => _networkLayers![layer];
        set => _networkLayers![layer] = value;
    }

    public Neuron this[int layer, int neuron]
    {
        get => _networkLayers![layer].GetNeurons()[neuron];
        set => _networkLayers![layer].SetNeuron(neuron, value);
    }

    public double this[int layer, int neuron, int weightOrBias]
    {
        get => weightOrBias < 0
            ? _networkLayers![layer].GetNeurons()[neuron].GetBias()
            : _networkLayers![layer].GetNeurons()[neuron].GetWeights()[weightOrBias];
        set
        {
            if (weightOrBias < 0)
                _networkLayers![layer].GetNeurons()[neuron].SetBias(value);
            else
                _networkLayers![layer].GetNeurons()[neuron].GetWeights()[weightOrBias] = value;
        }
    }

    // Network Construction
    
    public void CreateInputLayer(ushort numberOfNeurons, Neuron.ActivationType activation)
    {
        _inputLayer = new Layer(0, Layer.LayerType.Input);
        _inputLayer.Instantiate(numberOfNeurons, numberOfNeurons, activation);
        _networkLayers![0] = _inputLayer;
    }

    public void CreateOutputLayer(ushort numberOfNeurons, Neuron.ActivationType activation)
    {
        _outputLayer = new Layer((ushort)(_layerCount - 1)!, Layer.LayerType.Output);
        _outputLayer.Instantiate(numberOfNeurons, _networkLayers![(int)(_layerCount - 1)!].GetSize(), activation);
        _networkLayers![(int)(_layerCount - 1)!] = _outputLayer;
    }

    public void CreateHiddenLayers(ushort numberOfNeurons, Neuron.ActivationType activationType)
    {
        Layer afterInputLayer = new Layer(1, Layer.LayerType.Hidden);
        afterInputLayer.Instantiate(numberOfNeurons, _networkLayers![0].GetSize(), activationType);
        _networkLayers![1] = afterInputLayer;
        for (int i = 2; i < _layerCount - 1; i++)
        {
            Layer hiddenLayer = new Layer((ushort)i, Layer.LayerType.Hidden);
            hiddenLayer.Instantiate(numberOfNeurons, numberOfNeurons, activationType);
        }
    }
    
    public void CreateLayer(int identifier, ushort numberOfNeurons, Neuron.ActivationType activationType)
    {
        Layer layer = new Layer((ushort)(_layerCount - 1)!, identifier == 0 ? Layer.LayerType.Input : identifier == _layerCount - 1 ? Layer.LayerType.Output : Layer.LayerType.Hidden);
        layer.Instantiate(numberOfNeurons, _networkLayers![identifier - 1].GetSize(), activationType);
        _networkLayers![(int)(_layerCount - 1)!] = layer;
    }

    // Network Processing
    
    public double[] Process(string filePath)
    {
        string[] data = File.ReadAllLines(filePath);
        double[] results = new double[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            List<double> inputs = data[i].Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToList();
            foreach (var layer in _networkLayers!)
                inputs = layer.Process(inputs);
            results[i] = inputs[0];
        }

        return results;
    }

    private double ProcessSingle(string input)
    {
        List<double> outputs = input.Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToList();
        foreach (var layer in _networkLayers!)
            outputs = layer.Process(outputs);
        return outputs[0];
    }

    // Evaluation
    public double Accuracy(string filePath)
    {
        var data = File.ReadAllLines(filePath);
        double totalError = data.Sum(line =>
        {
            var parts = line.Split(';');
            var input = parts[0];
            var target = double.Parse(parts[1], CultureInfo.InvariantCulture);
            return Math.Abs(target - ProcessSingle(input));
        });
        return (1 - totalError / data.Length) * 100;
    }

    public void AccuracyWithPrint(string filePath)
    {
        string[] data = File.ReadAllLines(filePath);
        double totalError = 0;

        foreach (var line in data)
        {
            string[] parts = line.Split(";");
            string input = parts[0];
            double target = double.Parse(parts[1], CultureInfo.InvariantCulture);
            double predicted = ProcessSingle(input);
            Console.WriteLine($"Input: {input} Expected: {target} Predicted: {predicted}");
            totalError += Math.Abs(target - predicted);
        }

        Console.WriteLine($"Accuracy: {100 * (1 - totalError / data.Length)}%");
    }
    
    // Optimization
    
    public void Optimize(string trainingDataPath, ushort totalEpochs)
    {
        // Read all training samples from file
        string[] trainingSamples = File.ReadAllLines(trainingDataPath);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            foreach (string trainingSample in trainingSamples)
            {
                // Initialize gradient storage for each layer's parameters
                List<double[][]> weightGradientsPerLayer = new List<double[][]>();
                List<double[]> biasGradientsPerLayer = new List<double[]>();
                // Prepare gradient matrices for each layer
                foreach (Layer layer in _networkLayers!)
                {
                    int neuronsInLayer = layer.GetSize();
                    double[][] layerWeightGradients = new double[neuronsInLayer][];
                    double[] layerBiasGradients = new double[neuronsInLayer];
                    for (int neuronIndex = 0; neuronIndex < neuronsInLayer; neuronIndex++)
                    {
                        Neuron currentNeuron = layer.GetNeurons()[neuronIndex];
                        layerWeightGradients[neuronIndex] = new double[currentNeuron.GetDimensions()];
                        layerBiasGradients[neuronIndex] = 0;
                    }

                    weightGradientsPerLayer.Add(layerWeightGradients);
                    biasGradientsPerLayer.Add(layerBiasGradients);
                }

                // Parse input features and expected output from sample
                string[] sampleParts = trainingSample.Split(';');
                double[] inputFeatures = sampleParts[0].Split(',')
                    .Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                double expectedOutput = double.Parse(sampleParts[1], CultureInfo.InvariantCulture);
                // Forward pass storage for activation calculations
                List<List<NeuronActivationRecord>> forwardPassRecords = new List<List<NeuronActivationRecord>>();
                List<double> layerInputs = new List<double>(inputFeatures);
                // Forward propagation through each layer
                foreach (Layer layer in _networkLayers)
                {
                    List<NeuronActivationRecord> layerRecords = new List<NeuronActivationRecord>();
                    List<double> layerOutputs = new List<double>();
                    foreach (Neuron neuron in layer.GetNeurons())
                    {
                        // Retrieve inputs from parent layer
                        double[] neuronInputs = neuron.GetParents().Select(parentIndex => layerInputs[parentIndex])
                            .ToArray();
                        // Calculate weighted sum (z)
                        double weightedSum = neuron.GetBias();
                        for (int i = 0; i < neuronInputs.Length; i++)
                            weightedSum += neuron.GetWeights()[i] * neuronInputs[i];
                        // Apply activation function and calculate derivative
                        double activationOutput = 0;
                        double activationDerivative = 0;
                        switch (neuron.GetActivation())
                        {
                            case Neuron.ActivationType.Sigmoid:
                                activationOutput = Functions.Sigmoid(weightedSum);
                                activationDerivative = activationOutput * (1 - activationOutput);
                                break;
                            case Neuron.ActivationType.Linear:
                                activationOutput = weightedSum;
                                activationDerivative = 1;
                                break;
                            case Neuron.ActivationType.RElu:
                                activationOutput = Math.Max(0, weightedSum);
                                activationDerivative = weightedSum > 0 ? 1 : 0;
                                break;
                            case Neuron.ActivationType.Tanh:
                                activationOutput = Math.Tanh(weightedSum);
                                activationDerivative = 1 - activationOutput * activationOutput;
                                break;
                            case Neuron.ActivationType.AND:
                                activationOutput = weightedSum * weightedSum / 4 + weightedSum / 2 - 1;
                                activationDerivative = (weightedSum + 1) / 2;
                                break;
                            case Neuron.ActivationType.NAND:
                                activationOutput = -weightedSum * weightedSum / 4 - weightedSum / 2 + 1;
                                activationDerivative = -(weightedSum + 1) / 2;
                                break;
                            case Neuron.ActivationType.OR:
                                activationOutput = -weightedSum * weightedSum / 4 + weightedSum / 2 + 1;
                                activationDerivative = -(weightedSum - 1) / 2;
                                break;
                            case Neuron.ActivationType.NOR:
                                activationOutput = weightedSum * weightedSum / 4 - weightedSum / 2 - 1;
                                activationDerivative = (weightedSum - 1) / 2;
                                break;
                            case Neuron.ActivationType.EX:
                                activationOutput = -weightedSum * weightedSum / 2 + 1;
                                activationDerivative = -weightedSum;
                                break;
                            case Neuron.ActivationType.NEX:
                                activationOutput = weightedSum * weightedSum / 2 - 1;
                                activationDerivative = weightedSum;
                                break;
                        }

                        layerOutputs.Add(activationOutput);
                        layerRecords.Add(new NeuronActivationRecord
                        {
                            InputValues = neuronInputs,
                            WeightedSum = weightedSum,
                            ActivationOutput = activationOutput,
                            ActivationDerivative = activationDerivative
                        });
                    }

                    forwardPassRecords.Add(layerRecords);
                    layerInputs = layerOutputs;
                }

                double networkOutput = layerInputs[0];
                double lossGradient = 0;
                switch (_lossFunction)
                {
                    case LossFunction.CrossEntropy:
                        switch (_networkLayers[(int)(_layerCount - 1)!].GetNeurons()[0].GetActivation())
                        {
                            case Neuron.ActivationType.Sigmoid:
                                lossGradient = networkOutput - expectedOutput;
                                break;
                        }

                        break;
                    case LossFunction.MSE:
                        switch (_networkLayers[(int)(_layerCount - 1)!].GetNeurons()[0].GetActivation())
                        {
                            case Neuron.ActivationType.Linear:
                                lossGradient = networkOutput - expectedOutput;
                                break;
                        }

                        break;
                }

                // Backward propagation through layers
                double[] nextLayerDeltas = [lossGradient];
                for (int layerIndex = _networkLayers.Count() - 1; layerIndex >= 0; layerIndex--)
                {
                    Layer currentLayer = _networkLayers[layerIndex];
                    List<NeuronActivationRecord> layerActivationRecords = forwardPassRecords[layerIndex];
                    double[] currentLayerDeltas = new double[currentLayer.GetSize()];
                    for (int neuronIndex = 0; neuronIndex < currentLayer.GetSize(); neuronIndex++)
                    {
                        Neuron currentNeuron = currentLayer.GetNeurons()[neuronIndex];
                        NeuronActivationRecord activationRecord = layerActivationRecords[neuronIndex];
                        // Calculate delta for current neuron
                        double delta;
                        if (layerIndex == _networkLayers.Count() - 1) // Output layer
                            delta = nextLayerDeltas[neuronIndex];
                        else // Hidden layers
                        {
                            delta = 0;
                            Layer downstreamLayer = _networkLayers[layerIndex + 1];
                            // Sum contributions to error from downstream neurons
                            foreach (Neuron downstreamNeuron in downstreamLayer.GetNeurons())
                                for (int parentIndex = 0;
                                     parentIndex < downstreamNeuron.GetParents().Length;
                                     parentIndex++)
                                    if (downstreamNeuron.GetParents()[parentIndex] == neuronIndex)
                                        delta += downstreamNeuron.GetWeights()[parentIndex] *
                                                 nextLayerDeltas[
                                                     Array.IndexOf(downstreamLayer.GetNeurons(), downstreamNeuron)];
                            delta *= activationRecord.ActivationDerivative;
                        }

                        currentLayerDeltas[neuronIndex] = delta;
                        // Accumulate weight gradients
                        for (int weightIndex = 0; weightIndex < currentNeuron.GetDimensions(); weightIndex++)
                            weightGradientsPerLayer[layerIndex][neuronIndex][weightIndex] +=
                                delta * activationRecord.InputValues[weightIndex];
                        // Accumulate bias gradient
                        biasGradientsPerLayer[layerIndex][neuronIndex] += delta;
                    }

                    nextLayerDeltas = currentLayerDeltas;
                }

                // Update network parameters using accumulated gradients
                for (int layerIndex = 0; layerIndex < _networkLayers.Count(); layerIndex++)
                {
                    Layer currentLayer = _networkLayers[layerIndex];
                    for (int neuronIndex = 0; neuronIndex < currentLayer.GetSize(); neuronIndex++)
                    {
                        Neuron currentNeuron = currentLayer.GetNeurons()[neuronIndex];
                        // Update bias with learning rate
                        currentNeuron.SetBias(currentNeuron.GetBias() -
                                              (double)_learningRate! * biasGradientsPerLayer[layerIndex][neuronIndex]);
                        // Update weights with learning rate
                        for (int weightIndex = 0; weightIndex < currentNeuron.GetDimensions(); weightIndex++)
                            currentNeuron.GetWeights()[weightIndex] -= (double)_learningRate * weightGradientsPerLayer[layerIndex][neuronIndex][weightIndex];
                    }
                }
            }
            _learningRate /= 1.00005;
        }
    }

    public void RandomizeAndOptimize(ushort range, ushort attempts, ushort epochs, string filePath)
    {
        (double[], double)[][] originalSettings = StoreSettings();
        Randomize((ushort)(range * 2));
        (double[], double)[][] bestSettings = StoreSettings();
        double best = Accuracy(filePath);
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            Randomize((ushort)(range * 2));
            SetLearningRate(0.1);
            Optimize(filePath, epochs);
            double cur = Accuracy(filePath);
            if (cur > best)
            {
                best = cur;
                bestSettings = StoreSettings();
            }
        }
        ExtractSettings(bestSettings);
        double curPerf = Accuracy(filePath); 
        ExtractSettings(originalSettings);
        double originalPerf = Accuracy(filePath);
        if (originalPerf < curPerf)
            ExtractSettings(bestSettings);
    }

    private class NeuronActivationRecord
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
        sb.Append($"{_name};{_type};{_layerCount}\n");
        foreach (Layer layer in _networkLayers!)
            sb.Append(layer);
        sb.Remove(sb.Length - 1, 1);
        sb.Append($"{_lossFunction};{_learningRate?.ToString(CultureInfo.InvariantCulture)}");
        return sb.ToString();
    }

    public void SaveToFile(string filePath) => File.WriteAllText(filePath, ToString());

    /*private void LoadFromFile(string filePath)
    {
        string[] save = File.ReadAllLines(filePath);
        string[] header = save[0].Split(';');
        if (header.Length != 4) throw new FormatException("Invalid file header");
        _name = header[0];
        _lossFunction = Enum.Parse<LossFunction>(header[1]);
        _learningRate = double.Parse(header[2], CultureInfo.InvariantCulture);
        _layerCount = ushort.Parse(header[3]);
        _networkLayers = new Layer[(int)_layerCount];
        int linePointer = 1;
        for (int i = 0; i < _layerCount; i++)
        {
            if (linePointer >= save.Length)
                throw new FormatException($"Missing layer data for layer {i}");
            ushort neuronsInLayer = ushort.Parse(save[linePointer]);
            linePointer++;
            Layer currentLayer = new Layer((ushort)i, neuronsInLayer);
            for (int k = 0; k < neuronsInLayer; k++)
            {
                if (linePointer >= save.Length)
                    throw new FormatException($"Missing neuron data for layer {i} neuron {k}");
                string[] neuronData = save[linePointer].Split(';');
                if (neuronData.Length != 5)
                    throw new FormatException($"Invalid neuron data at line {linePointer}");
                ushort dimensions = ushort.Parse(neuronData[0]);
                double[] weights = neuronData[1].Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                double bias = double.Parse(neuronData[2], CultureInfo.InvariantCulture);
                string activation = neuronData[3];
                ushort[] parents = neuronData[4].Split(',').Select(ushort.Parse).ToArray();
                currentLayer.SetNeuron(k, new Neuron((ushort)k, (ushort)i, dimensions, weights, bias, activation, parents));
                linePointer++;
            }
            _networkLayers[i] = currentLayer;
        }
    }*/

    // Randomization / Testing

    public void Randomize(ushort range)
    {
        Random rand = new Random();
        foreach (var layer in _networkLayers!)
        {
            foreach (var neuron in layer.GetNeurons())
            {
                for (int i = 0; i < neuron.GetDimensions(); i++)
                    neuron.GetWeights()[i] = (rand.NextDouble() - 0.5) * range;
                neuron.SetBias((rand.NextDouble() - 0.5) * range);
            }
        }
    }

    private (double[], double)[][] StoreSettings()
    {
        return _networkLayers!
            .Select(layer => layer.GetNeurons()
                .Select(neuron => (
                    neuron.GetWeights().ToArray(),
                    neuron.GetBias()
                )).ToArray())
            .ToArray();
    }

    private void ExtractSettings((double[], double)[][] settings)
    {
        for (int l = 0; l < _networkLayers!.Length; l++)
        {
            for (int n = 0; n < _networkLayers[l].GetSize(); n++)
            {
                _networkLayers[l].GetNeurons()[n].SetBias(settings[l][n].Item2);
                for (int w = 0; w < _networkLayers[l].GetNeurons()[n].GetDimensions(); w++)
                    _networkLayers[l].GetNeurons()[n].GetWeights()[w] = settings[l][n].Item1[w];
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
}
