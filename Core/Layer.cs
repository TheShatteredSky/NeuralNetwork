namespace NeuralNetwork.Core;

public class Layer
{
    private ushort _size;
    private Node[] _nodes;
    private LayerType _type;
    
    public Node this[int node]
    {
        get => _nodes[node];
        set => _nodes[node] = value;
    }
    
    public Layer(ushort identifier, LayerType type)
    {
        if (identifier == 0 && type != LayerType.Input) throw new ArgumentException("Layer type of the first Layer cannot be non-input.");
        _type = type;
        _size = 0;
        _nodes = [];
    }
    
    internal void InstantiateCustom(ushort size)
    {
        _size = size;
        _nodes = new Node[size];
    }
    
    internal void Instantiate(ushort size, ushort previousLayerSize, ActivationType activationType)
    {
        _size = size;
        _nodes = new Node[size];
        for (int i = 0; i < _nodes.Length; i++)
            _nodes[i] = new Node((ushort)(_type == LayerType.Input ? 1 : previousLayerSize), activationType, _type == LayerType.Input ? [(ushort)i] : null);
    }
    
    public ushort GetSize() => _size;
    
    public Node[] GetNodes() => _nodes;
    
    public LayerType GetLayerType() => _type;
    
    internal double[] Process(double[] inputs)
    {
        double[] results = new double[_size];
        bool softmaxLayer = _nodes[0].GetActivation() == ActivationType.Softmax;
        if (softmaxLayer) results = SoftmaxOutputs(WeightedSums(inputs));
        else
        {
            Parallel.For(0, _size, n =>
            {
                double[] nodeInputs = NodeInputs(inputs, n);
                results[n] = _nodes[n].Process(nodeInputs);
            });
        }
        return results;
    }

    internal double[] WeightedSums(double[] inputs)
    {
        double[] results = new double[_size];
        Parallel.For(0, _size, n =>
        {
            results[n] = _nodes[n].WeightedSum(NodeInputs(inputs, n)); 
        });
        return results;
    }

    internal double[] NodeInputs(double[] inputs, int n)
    {
        ushort[]? parents = _nodes[n].GetParents();
        return parents == null ? inputs : parents.Select(x => inputs[x]).ToArray();
    }

    internal double[] SoftmaxOutputs(double[] weightedSums)
    {
        double max = weightedSums.Max();
        double sumExp = 0;
        double[] softmaxOutputs = new double[weightedSums.Length];
        for (int i = 0; i < weightedSums.Length; i++)
        {
            softmaxOutputs[i] = Math.Exp(weightedSums[i] - max);
            sumExp += softmaxOutputs[i];
        }
        for (int i = 0; i < weightedSums.Length; i++)
            softmaxOutputs[i] /= sumExp;
        return softmaxOutputs;
    }

    /// <summary>
    /// The formatted string representing this Layer.
    /// ⚠ This is a custom format, not JSON.
    /// </summary>
    /// <returns>A string representing this Layer.</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append(_type + ";" + _size + "\n");
        foreach (Node node in _nodes)
            sb.Append(node);
        return sb.ToString();
    }
}