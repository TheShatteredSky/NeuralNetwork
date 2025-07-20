namespace NeuralNetwork.Core;

/// <summary>
/// A Layer instance.
/// </summary>
public class Layer
{
    private ushort _size;
    private Node[] _nodes;
    private LayerType _type;
    
    /// <summary>
    /// Indexer for this Layer's Nodes.
    /// ⚠ This method returns a reference to the actual Node this Layer uses, modifying it will modify the that of Network too.
    /// </summary>
    /// <param name="nodeIndex">The index of the Node.</param>
    public Node this[int nodeIndex]
    {
        get => _nodes[nodeIndex];
        set => _nodes[nodeIndex] = value;
    }

    /// <summary>
    /// Indexer for the parameters of this Layer's Nodes.
    /// </summary>
    /// <param name="nodeIndex">The index of the Node.</param>
    /// <param name="paramaterIndex">The index of the parameter, 0 to dimensions - 1 will return the specified weight, while dimensions will return the bias.</param>
    public double this[int nodeIndex, int paramaterIndex]
    {
        get => _nodes[nodeIndex][paramaterIndex];
        set => _nodes[nodeIndex][paramaterIndex] = value;
    }
    
    /// <summary>
    /// Creates a new Layer of the specified type.
    /// </summary>
    /// <param name="type">This Layer's type.</param>
    /// <exception cref="ArgumentException"></exception>
    public Layer(LayerType type)
    {
        _type = type;
        _size = 0;
        _nodes = [];
    }
    
    /// <summary>
    /// Instantiates this Layer's Node array without creating its Nodes.
    /// </summary>
    /// <param name="size">This Layer's size.</param>
    internal void Instantiate(ushort size)
    {
        _size = size;
        _nodes = new Node[size];
    }
    
    /// <summary>
    /// Instantiates this Layer's Node array and creates its Nodes.
    /// </summary>
    /// <param name="size">This Layer's size.</param>
    /// <param name="previousLayerSize">The previous Layer's size. Note: This is needed because all Nodes are fully connected by default.</param>
    /// <param name="activationType">The activation function for the Nodes of this Layer.</param>
    internal void Instantiate(ushort size, ushort previousLayerSize, ActivationType activationType)
    {
        _size = size;
        _nodes = new Node[size];
        for (int i = 0; i < _nodes.Length; i++)
            _nodes[i] = new Node((ushort)(_type == LayerType.Input ? 1 : previousLayerSize), activationType, _type == LayerType.Input ? [(ushort)i] : null);
    }
    
    /// <summary>
    /// Fetches the size (number of Nodes) of this Layer.
    /// </summary>
    /// <returns>The size of this Layer.</returns>
    public ushort GetSize() => _size;
    
    /// <summary>
    /// Fetches this Layer's Node array.
    /// ⚠ This method returns a reference to the actual Node array this Layer uses, modifying it will modify the that of this Layer too.
    /// </summary>
    /// <returns>The Node array of this Layer.</returns>
    public Node[] GetNodes() => _nodes;
    
    /// <summary>
    /// Fetches this Layer's type.
    /// </summary>
    /// <returns>The type of this Layer.</returns>
    public LayerType GetLayerType() => _type;
    
    /// <summary>
    /// Computes the outputs of this Layer's Nodes.
    /// </summary>
    /// <param name="inputs">The Nodes' inputs.</param>
    /// <returns>The outputs of this Layer.</returns>
    //Note: At no point should the input array be modified or returned.
    //TODO: Removed parallel processing because of race condition worries, recheck it later.
    internal double[] Process(double[] inputs)
    {
        if (_nodes[0].GetActivation() == ActivationType.Softmax) return SoftmaxOutputs(WeightedSums(inputs));
        double[] results = new double[_size];
            for (int n = 0; n < _size; n++)
                results[n] = _nodes[n].Process(inputs);
        return results;
    }

    /// <summary>
    /// Computes the weighted sums of this Layer's Nodes.
    /// </summary>
    /// <param name="inputs">The Nodes' inputs.</param>
    /// <returns>The weighted sums of this Layer.</returns>
    //Note: At no point should the input array be modified or returned.
    //TODO: Removed parallel processing because of race condition worries, recheck it later.
    internal double[] WeightedSums(double[] inputs)
    {
        double[] results = new double[_size];
        for (int n = 0; n < _size; n++)
            results[n] = _nodes[n].WeightedSum(inputs); 
        return results;
    }

    /// <summary>
    /// Computes the outputs for the Softmax activation.
    /// </summary>
    /// <param name="weightedSums">The weighted sums of this Layer's Nodes.</param>
    /// <returns>The outputs of this Layer's Nodes.</returns>
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