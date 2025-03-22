namespace NeuralNetwork;

public class Layer
{
    
    private readonly ushort _identifier;
    private ushort _size;
    private Neuron[] _neurons; 
    
    public Layer(ushort identifier, ushort size)
    {
        _identifier = identifier;
        _size = size;
        _neurons = new Neuron[size];
    }
    
    public ushort GetSize() => _size;
    
    public Neuron[] GetNeurons() => _neurons;
    
    public ushort GetLayerIdentifier() => _identifier;
    
    public void SetNeuron(int i, Neuron neuron) => _neurons[i] = neuron;

    public void IncreaseLayerSize()
    {
        Neuron[] newNeurons = new Neuron[_size + 1];
        Array.Copy(_neurons, 0, newNeurons, 0, _neurons.Length);
        _neurons = newNeurons;
        _size++;
    }
    
    public List<double> Process(List<double> factors)
    {
        double[] resultFactors = new double[_size];
        for (int i = 0; i < _size; ++i)
        {
            ushort[] parents = _neurons[i].GetParents();
            double[] inputs = new double[parents.Length];
            for (int j = 0; j < parents.Length; ++j)
                inputs[j] = factors[parents[j]];
            resultFactors[i] = _neurons[i].Process(inputs);
        }
        return resultFactors.ToList();
    }
}