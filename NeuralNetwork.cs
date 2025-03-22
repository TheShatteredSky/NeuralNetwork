namespace NeuralNetwork;
using System.Globalization;
using System;
using System.IO;
using System.Text;

public class NeuralNetwork
{
    private readonly string? _name;
    private ushort? _layerCount;
    private Layer[]? _networkLayers;
    private LossFunction? _lossFunction;
    private double? _learningRate;
    private double? _seed;
    private Random? _random;
    
    //Empty constructor if I need it for whatever reason.
    public NeuralNetwork()
    {
        
    }
    
    public NeuralNetwork(string name, ushort layerCount, string lossFunction)
    {
        _name = name;
        _layerCount = layerCount;
        _networkLayers = new Layer[(ushort)_layerCount];
        _random = new Random();
        _seed = _random.NextDouble();
        _lossFunction = Enum.Parse<LossFunction>(lossFunction, true);
    }

    public NeuralNetwork(string filePath)
    {
        string[] save = File.ReadAllLines(filePath);
        string[] header = save[0].Split(';');
        if (header.Length != 5) throw new FormatException("Invalid file header");
        _name = header[0];
        _lossFunction = Enum.Parse<LossFunction>(header[1]);
        _learningRate = double.Parse(header[2], CultureInfo.InvariantCulture);
        _seed = double.Parse(header[3], CultureInfo.InvariantCulture);
        _layerCount = ushort.Parse(header[4]);
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
    }

    public void SetLearningRate(double learningRate) => _learningRate = learningRate;
    public ushort GetLayerCount() => (ushort)_layerCount!;
    public Layer[] GetLayers() => _networkLayers!;
    public void SetLayer(ushort i, Layer l) => _networkLayers![i] = l;
    public void SetNeuron(ushort layer, ushort neuron, ushort dimensions, string activation, ushort[] parents) => _networkLayers![layer].GetNeurons()[neuron] = new Neuron(layer, neuron, dimensions, activation, parents);
    public void SetLoss(string loss)
    {
        switch (loss)
        {
            case "CrossEntropy":
                _lossFunction = LossFunction.CrossEntropy;
                break;
        }
    }
    
    private enum LossFunction
    {
        CrossEntropy,
        MSE,
    }

    public void AddLayer(ushort numberOfNeurons)
    {
        _layerCount++;
        _networkLayers![(int)(_layerCount - 1)!] = new Layer((ushort)(_layerCount - 1)!, numberOfNeurons);
    }

    public void SetSeed(int seed)
    {
        _random = new Random(seed);
    }

    public void CreateFirstLayer(ushort numberOfNeurons)
    {
        _networkLayers = new Layer[1];
        _networkLayers[0] = new Layer(0, numberOfNeurons);
        _layerCount = 1;
    }

    public void AddNeuron(ushort neuron, ushort layer, ushort dimensions, string activation, ushort[] parents)
    {
        _networkLayers![layer].IncreaseLayerSize();
        _networkLayers[layer].SetNeuron(neuron, new Neuron(neuron, layer, dimensions, activation, parents));
    }
    
    public double[] Process(string filePath)
    {
        string[] data = File.ReadAllLines(filePath);
        double[] results = new double[data.Length];
    
        for (int i = 0; i < data.Length; i++)
        {
            List<double> outputOfLayer = data[i].Split(",").Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToList();
            for (int j = 0; j < _layerCount; j++)
                outputOfLayer = _networkLayers![j].Process(outputOfLayer);
            results[i] = outputOfLayer[0];
        }
        return results;
    }

    private double ProcessSingle(string data)
    {
        List<double> outputOfLayer = data.Split(",").Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToList();
        for (int j = 0; j < _layerCount; j++)
            outputOfLayer = _networkLayers![j].Process(outputOfLayer);
        return outputOfLayer[0];
    }

    public void AddNeuron(ushort layer, ushort dimensions, string activation, ushort[] parents)
    {
        Neuron[] newNeurons = new Neuron[_networkLayers![layer].GetSize() + 1];
        Array.Copy(_networkLayers[layer].GetNeurons(), newNeurons, _networkLayers[layer].GetSize());
        newNeurons[_networkLayers[layer].GetSize()] = new Neuron(_networkLayers[layer].GetSize(), layer, dimensions, activation, parents);
    }

    public double Accuracy(string filePath)
    {
        string[] data = File.ReadAllLines(filePath);
        double total = 0;
        foreach (var line in data)
        {
            string[] parts = line.Split(";");
            string input = parts[0];
            double target = double.Parse(parts[1], CultureInfo.InvariantCulture); // Add invariant culture
            total += Math.Abs(target - ProcessSingle(input));
        }
        return (1 - total / data.Length) * 100;
    }

    public void AccuracyWithPrint(string filePath)
    {
        string[] data = File.ReadAllLines(filePath);
        double total = 0;
        foreach (var line in data)
        {
            string[] parts = line.Split(";");
            string input = parts[0];
            double target = double.Parse(parts[1], CultureInfo.InvariantCulture); // Add invariant culture
            double predicted = ProcessSingle(input);
            Console.WriteLine($"Input: {input} Expected: {target} Predicted: {predicted}");
            total += Math.Abs(target - predicted);
        }
        Console.WriteLine($"Accuracy: {100 * (1 - total / data.Length)}%");
    }
    
    public new string ToString()
    {
        StringBuilder sb = new StringBuilder(); 
        sb.Append(_name + ";" + _lossFunction + ";" + _learningRate + ";" + _seed + ";" + _layerCount + ";");
        for (int i = 0; i < _layerCount; i++)
        {
            sb.Append(_networkLayers![i].GetSize() + ";");
            for (int j = 0; j < _networkLayers[i].GetSize(); j++)
            {
                sb.Append(_networkLayers[i].GetNeurons()[j].GetDimensions() + ";");
                for (int k = 0; k < _networkLayers[i].GetNeurons()[j].GetDimensions(); k++)
                    sb.Append(_networkLayers[i].GetNeurons()[j].GetWeights()[k] + ";");
                sb.Append(_networkLayers[i].GetNeurons()[j].GetBias() + ";" + _networkLayers[i].GetNeurons()[j].GetActivation() + ";");
                for (int k = 0; k < _networkLayers[i].GetNeurons()[j].GetDimensions(); k++)
                    sb.Append(_networkLayers[i].GetNeurons()[j].GetParents()[k] + ";");
            }
        }
        return sb.ToString();
    }

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
                double[] inputFeatures = sampleParts[0].Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
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
                        double[] neuronInputs = neuron.GetParents().Select(parentIndex => layerInputs[parentIndex]).ToArray();
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
                                activationOutput = -weightedSum * weightedSum / 4 + -weightedSum / 2 + 1;
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
                                for (int parentIndex = 0; parentIndex < downstreamNeuron.GetParents().Length; parentIndex++)
                                    if (downstreamNeuron.GetParents()[parentIndex] == neuronIndex)
                                        delta += downstreamNeuron.GetWeights()[parentIndex] * nextLayerDeltas[Array.IndexOf(downstreamLayer.GetNeurons(), downstreamNeuron)];
                            delta *= activationRecord.ActivationDerivative;
                        }
                        currentLayerDeltas[neuronIndex] = delta;
                        // Accumulate weight gradients
                        for (int weightIndex = 0; weightIndex < currentNeuron.GetDimensions(); weightIndex++)
                            weightGradientsPerLayer[layerIndex][neuronIndex][weightIndex] += delta * activationRecord.InputValues[weightIndex];
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
                        //Console.Write("Neuron #" + layerIndex + ";" + neuronIndex + " , bias went from " + currentNeuron.GetBias());
                        currentNeuron.SetBias(currentNeuron.GetBias() - (double)_learningRate! * biasGradientsPerLayer[layerIndex][neuronIndex]);
                        //Console.WriteLine(" to " + currentNeuron.GetBias());
                        // Update weights with learning rate
                        for (int weightIndex = 0; weightIndex < currentNeuron.GetDimensions(); weightIndex++)
                        {
                            //Console.Write("Neuron #" + layerIndex + ";" + neuronIndex + " , weight #" + weightIndex + " went from " + currentNeuron.GetWeights()[weightIndex]);
                            currentNeuron.GetWeights()[weightIndex] -= (double)_learningRate * weightGradientsPerLayer[layerIndex][neuronIndex][weightIndex];
                            //Console.WriteLine(" to " + currentNeuron.GetWeights()[weightIndex]);
                        }
                        /*if (currentNeuron.GetActivation() == Neuron.ActivationType.AND ||
                            currentNeuron.GetActivation() == Neuron.ActivationType.NAND ||
                            currentNeuron.GetActivation() == Neuron.ActivationType.OR ||
                            currentNeuron.GetActivation() == Neuron.ActivationType.NOR ||
                            currentNeuron.GetActivation() == Neuron.ActivationType.EX ||
                            currentNeuron.GetActivation() == Neuron.ActivationType.NEX)
                        {
                            currentNeuron.SetBias(0);
                            currentNeuron.GetWeights()[0] = 1;
                            currentNeuron.GetWeights()[1] = 1;
                        }*/
                    }
                }
            }
            _learningRate /= 1.00005;
        }
    }


    private class NeuronActivationRecord
    {
        public required double[] InputValues { get; init; } // Inputs received from previous layer
        public double WeightedSum { get; set; } // z = sum(weights * inputs) + bias
        public double ActivationOutput { get; set; } // a = activation(z)
        public double ActivationDerivative { get; init; } // da/dz derivative at z
    }

    public void RandomizeAndOptimize(ushort range, ushort attempts, ushort epochs, string filePath)
    {
        (double[], double)[][] originalSettings = StoreSettings();
        Randomize((ushort)(range*2));
        (double[], double)[][] bestSettings = StoreSettings();
        double best = Accuracy(filePath);
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            Randomize((ushort)(range*2));
            _learningRate = 0.1;
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

    private void Randomize(ushort range)
    {
        Random rand = new Random();
        for (int layer = 0; layer < _networkLayers!.Count(); layer++)
        {
            Layer currentLayer = _networkLayers![layer];
            for (int neuron = 0; neuron < currentLayer.GetSize(); neuron++)
            {
                Neuron currentNeuron = currentLayer.GetNeurons()[neuron];
                double[] weights = currentNeuron.GetWeights();
                for (int weight = 0; weight < currentNeuron.GetDimensions(); weight++)
                    weights[weight] = (rand.NextDouble() - 0.5) * range;
                currentNeuron.SetBias((rand.NextDouble() - 0.5) * range);
            }
        }
    }
    
    private (double[], double)[][] StoreSettings()
    {
        (double[], double)[][] storage = new (double[], double)[_networkLayers!.Count()][];
        for (int layer = 0; layer < _networkLayers!.Count(); layer++)
        {
            Layer currentLayer = _networkLayers![layer];
            storage[layer] = new (double[], double)[currentLayer.GetSize()]; 
            for (int neuron = 0; neuron < currentLayer.GetSize(); neuron++)
            {
                Neuron currentNeuron = currentLayer.GetNeurons()[neuron];
                storage[layer][neuron].Item1 = new double[currentNeuron.GetDimensions()];
                for (int weight = 0; weight < currentNeuron.GetDimensions(); weight++)
                    storage[layer][neuron].Item1[weight] = currentNeuron.GetWeights()[weight];
                storage[layer][neuron].Item2 = currentNeuron.GetBias();
            }
        }
        return storage;
    }

    private void ExtractSettings((double[], double)[][] storage)
    {
        for (int layer = 0; layer < _networkLayers!.Count(); layer++)
        {
            for (int neuron = 0; neuron < _networkLayers![layer].GetSize(); neuron++)
            {
                for (int weight = 0; weight < _networkLayers![layer].GetNeurons()[neuron].GetDimensions(); weight++)
                    _networkLayers![layer].GetNeurons()[neuron].GetWeights()[weight] = storage[layer][neuron].Item1[weight];
                _networkLayers![layer].GetNeurons()[neuron].SetBias(storage[layer][neuron].Item2);
            }
        }
    }

    public void SaveToFile(string filePath)
    {
        List<string> saveData = [$"{_name};{_lossFunction};{_learningRate?.ToString(CultureInfo.InvariantCulture)};{_seed?.ToString(CultureInfo.InvariantCulture)};{_layerCount}"];
        foreach (Layer layer in _networkLayers!)
        {
            saveData.Add(layer.GetSize().ToString());
            foreach (Neuron neuron in layer.GetNeurons())
            {
                string weights = string.Join(",", neuron.GetWeights().Select(w => w.ToString(CultureInfo.InvariantCulture)));
                string parents = string.Join(",", neuron.GetParents());
                saveData.Add($"{neuron.GetDimensions()};" + $"{weights};" + $"{neuron.GetBias().ToString(CultureInfo.InvariantCulture)};" + $"{neuron.GetActivation()};" + $"{parents}");
            }
        }
        File.WriteAllLines(filePath, saveData);
    }
}
//A