using System.Text;

namespace NeuralNetwork;

public class Layer
{
    
    private readonly ushort _identifier;
    private ushort? _size;
    private Neuron[]? _neurons;
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
    }

    public void Instantiate(ushort size, ushort previousLayerSize, Neuron.ActivationType activationType)
    {
        _size = size;
        _neurons = new Neuron[size];
        for (int i = 0; i < _neurons.Length; i++)
        {
            _neurons[i] = new Neuron((ushort)i, _identifier, previousLayerSize, activationType, null, false);
        }
    }
    
    public ushort GetSize() => (ushort)_size!;
    
    public Neuron[] GetNeurons() => _neurons!;
    
    public ushort GetLayerIdentifier() => _identifier;
    
    public void SetNeuron(int i, Neuron neuron) => _neurons![i] = neuron;

    public void IncreaseLayerSize()
    {
        Neuron[] newNeurons = new Neuron[(int)_size! + 1];
        Array.Copy(_neurons!, 0, newNeurons, 0, _neurons!.Length);
        _neurons = newNeurons;
        _size++;
    }
    
    public List<double> Process(List<double> factors)
    {
        double[] resultFactors = new double[(int)_size!];
        for (int i = 0; i < _size; ++i)
        {
            
            double[] inputs;
            if (_neurons![i].HasParents())
            {
               inputs =  new double[factors.Count];
               for (int j = 0; j < factors.Count; ++j)
                   inputs[j] = factors[j];
            }
            else
            {
                ushort[] parents = _neurons![i].GetParents();
                inputs = new double[parents.Length]; 
                for (int j = 0; j < parents.Length; ++j)
                    inputs[j] = factors[parents[j]];
            }
            resultFactors[i] = _neurons[i].Process(inputs);
        }
        return resultFactors.ToList();
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append(_size + "\n");
        foreach (Neuron neuron in _neurons!)
            sb.Append(neuron);
        return sb.ToString();
    }
}