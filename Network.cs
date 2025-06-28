namespace NeuralNetwork;

using System;
using System.Text;

public class Network
{
    // Fields

    private string? _name;
    
    private ushort? _layerCount;
    private Layer[]? _networkLayers;
    

    // Constructors
    
    public Network(string? name)
    {
        _name = name ?? "NeuralNetwork";
    }
    

    //  Instantiation Methods

    public void Instantiate(int hiddenLayers)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
    }
    

    // Setters
    
    public void SetLayer(ushort i, Layer l) => _networkLayers![i] = l;
    public void SetNode(ushort layer, ushort node, ushort dimensions, Node.ActivationType activation, ushort[] parents) => _networkLayers![layer].GetNodes()[node] = new Node(layer, node, dimensions, activation, parents);
    public void SetName(string newName) => _name = newName;

    // Getters
    
    public string GetName() => _name!;
    public ushort GetLayerCount() => (ushort)_layerCount!;
    public Layer[] GetLayers() => _networkLayers!;

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
        Layer input = new Layer(0, Layer.LayerType.Input);
        input.Instantiate(numberOfNodes, 1, activation);
        _networkLayers![0] = input;
    }

    public void CreateOutputLayer(ushort numberOfNodes, Node.ActivationType activation)
    {
        Layer output = new Layer((ushort)(_layerCount - 1)!, Layer.LayerType.Output);
        output.Instantiate(numberOfNodes, _networkLayers![(int)(_layerCount - 2)!].GetSize(), activation);
        _networkLayers![(int)(_layerCount - 1)!] = output;
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
    
    public double[][] Process(double[][] features) {
        return features.Select(ProcessSingle).ToArray();
    }

    private double[] ProcessSingle(double[] features) {
        double[] current = features;
        foreach (var layer in _networkLayers!) 
            current = layer.Process(current);
        return current;
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
    
    public override string ToString()
    {
       StringBuilder sb = new StringBuilder();
        sb.Append($"{_name};{_layerCount}\n");
        foreach (Layer layer in _networkLayers!)
            sb.Append(layer);
        return sb.ToString();
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
                    node.GetWeights()[i] = NetworkUtilities.NextDouble(-range, range);
                node.SetBias(0);
            }
        }
    }
}
//