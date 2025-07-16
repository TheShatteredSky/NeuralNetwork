namespace NeuralNetwork.Core;

using System.Text;

public class Layer
{
    private readonly ushort _identifier;
    private ushort _size;
    private Node[] _nodes;
    private LayerType _type;

    public enum LayerType
    {
        Input,
        Hidden,
        Output,
    }
        
    public Layer(ushort identifier, LayerType type)
    {
        _identifier = identifier;
        _type = type;
        _size = 0;
        _nodes = [];
    }
    
    internal void InstantiateCustom(ushort size)
    {
        _size = size;
        _nodes = new Node[size];
    }
    
    internal void Instantiate(ushort size, ushort previousLayerSize, Node.ActivationType activationType)
    {
        _size = size;
        _nodes = new Node[size];
        ushort[] parents = new ushort[previousLayerSize];
        for (int i = 0; i < previousLayerSize; i++)
            parents[i] = (ushort)i;
        for (int i = 0; i < _nodes.Length; i++)
            _nodes[i] = new Node((ushort)i, _identifier, (ushort)(_identifier == 0 ? 1 : previousLayerSize), activationType, _identifier == 0 ? [(ushort)i] : parents);
    }
    
    public ushort GetSize() => (ushort)_size!;
    
    public Node[] GetNodes() => _nodes!;
    
    public ushort GetLayerIdentifier() => _identifier;
    
    public void SetNodes(int i, Node node) => _nodes![i] = node;
    
    public LayerType GetLayerType() => _type;
    
    internal double[] Process(double[] inputs)
    {
        double[] results = new double[_size];
        bool softmaxLayer = _nodes![0].GetActivation() == Node.ActivationType.Softmax;
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
            results[n] = _nodes![n].WeightedSum(NodeInputs(inputs, n)); 
        });
        return results;
    }

    internal double[] NodeInputs(double[] inputs, int n)
    {
        double[] nodeInputs;
        nodeInputs = _nodes![n].GetParentCount() == inputs.Length ? inputs : _nodes[n].GetParents().Select(parentIndex => inputs[parentIndex]).ToArray();
        return nodeInputs;
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

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append(_type + ";" + _size + "\n");
        foreach (Node node in _nodes!)
            sb.Append(node);
        return sb.ToString();
    }

    public Node this[int index]
    {
        get => _nodes![index];
        set => _nodes![index] = value;
    }
}
//