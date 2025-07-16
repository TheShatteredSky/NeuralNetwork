using NeuralNetwork.Optimizers;

namespace NeuralNetwork.Core;

using System;
using System.Text;
using Addons;

public class Network
{
    private string? _name;
    private ushort _layerCount;
    private Layer[] _networkLayers;

    public Network()
    {
        _name = null;
        _layerCount = 0;
        _networkLayers = [];
    }

    public void Instantiate(int hiddenLayers)
    {
        _networkLayers = new Layer[hiddenLayers + 2];
        _layerCount = (ushort)(hiddenLayers + 2);
    }

    public void SetLayer(ushort i, Layer l) => _networkLayers![i] = l;

    public void
        SetNode(ushort layer, ushort node, ushort dimensions, Node.ActivationType activation, ushort[] parents) =>
        _networkLayers![layer].GetNodes()[node] = new Node(layer, node, dimensions, activation, parents);

    public void SetName(string newName) => _name = newName;

    public string GetName() => _name!;
    public ushort GetLayerCount() => (ushort)_layerCount!;
    public Layer[] GetLayers() => _networkLayers!;

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
        Layer layer = new Layer((ushort)(_layerCount - 1)!,
            identifier == 0 ? Layer.LayerType.Input :
            identifier == _layerCount - 1 ? Layer.LayerType.Output : Layer.LayerType.Hidden);
        layer.Instantiate(numberOfNodes, _networkLayers![identifier - 1].GetSize(), activationType);
        _networkLayers![(int)(_layerCount - 1)!] = layer;
    }

    public double[][] Process(double[][] inputs)
    {
        return inputs.Select(ProcessSingle).ToArray();
    }

    internal double[] ProcessSingle(double[] inputs)
    {
        double[] current = inputs;
        foreach (var layer in _networkLayers!)
            current = layer.Process(current);
        return current;
    }

    public double AbsoluteLoss(double[][] inputs, double[][] outputs)
    {
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double currentError = 0;
            double[] predictions = ProcessSingle(inputs[i]);
            for (int j = 0; j < outputs[i].Length; j++)
                currentError += Math.Abs(outputs[i][j] - predictions[j]);
            currentError /= outputs[i].Length;
            totalError += currentError;
        }

        totalError /= inputs.Length;
        return totalError;
    }

    public double Loss(double[][] inputs, double[][] outputs, Optimizer.LossFunction lossFunction)
    {
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double[] predictions = ProcessSingle(inputs[i]);
            double sampleLoss = 0;

            switch (lossFunction)
            {
                case Optimizer.LossFunction.MSE:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.MSE(predictions[j], outputs[i][j]);
                    sampleLoss /= outputs[i].Length;
                    break;
                case Optimizer.LossFunction.BinaryCrossEntropy:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.BinaryCrossEntropy(predictions[j], outputs[i][j]);
                    break;
                case Optimizer.LossFunction.CategoricalCrossEntropy:
                    for (int j = 0; j < outputs[i].Length; j++)
                        sampleLoss += LossFunction.CategoricalCrossEntropy(predictions[j], outputs[i][j]);
                    break;
            }

            totalError += sampleLoss;
        }

        return totalError / inputs.Length; // Average over samples
    }

    public void AbsoluteLossWithPrint(double[][] inputs, double[][] outputs)
    {
        double totalError = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double currError = 0;
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < inputs[i].Length; j++)
                sb.Append(inputs[i][j] + ",");
            string input = sb.ToString();
            double[] predictions = ProcessSingle(inputs[i]);
            Console.WriteLine(
                $"Input: {input} Predicted: {string.Join(", ", predictions)} Expected: {string.Join(", ", outputs[i])}");
            for (int j = 0; j < outputs[i].Length; j++)
                currError += Math.Abs(outputs[i][j] - predictions[j]);
            currError /= outputs[i].Length;
            totalError += currError;
        }

        totalError /= inputs.Length;
        Console.WriteLine($"Loss: {100 * totalError}%");
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append($"{_name};{_layerCount}\n");
        foreach (Layer layer in _networkLayers!)
            sb.Append(layer);
        return sb.ToString();
    }

    public void Randomize(double range)
    {
        foreach (var layer in _networkLayers!)
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