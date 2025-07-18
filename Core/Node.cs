namespace NeuralNetwork.Core;

public class Node
{
   private readonly ushort _identifier;
   private readonly ushort _layerIdentifier;
   private ushort _dimensions;
   private double[] _weights;
   private double _bias;
   private ActivationType _activation;
   private ushort[] _parents;

   public double this[int param]
   {
       get => param < _dimensions ? _weights[param] : _bias;
       set
       {
           if (param < _dimensions)
               _weights[param] = value;
           else
               _bias = value;
       }
   }
   
   public Node(ushort identifier, ushort layerIdentifier, ushort dimensions, double[] weights, double bias, ActivationType activation, ushort[] parents)
   {
       _identifier = identifier;
       _layerIdentifier = layerIdentifier;
       if (dimensions != weights.Length) throw new ArgumentException("The number of dimensions does not match the number of weights.");
       _dimensions = dimensions;
       _activation = activation;
       _weights = weights;
       _bias = bias;
       if (parents.Length > dimensions) throw new ArgumentException("The given parents are more numerous than the node's dimensions.");
       _parents = parents;
   }
   
   public Node(ushort identifier, ushort layerIdentifier, ushort dimensions, ActivationType ac, ushort[] parents)
   {
       _identifier = identifier;
       _layerIdentifier = layerIdentifier;
       _dimensions = dimensions;
       _weights = new double[dimensions];
       for (int i = 0; i < _weights.Length; i++)
       {
           switch (_activation)
           {
               case ActivationType.RElu:
                   double std = Math.Sqrt(2.0 / dimensions);
                   _weights[i] = NetworkUtilities.NextDouble(-std, std);
                   break;
               case ActivationType.Sigmoid:
                   double range = Math.Sqrt(6.0 / (dimensions + 1));
                   _weights[i] = NetworkUtilities.NextDouble(-range, range);
                   break;
               default:
                   _weights[i] = 1;
                   break;
           }
       }
       _bias = 0;
       _activation = ac;
       if (parents.Length > dimensions) throw new ArgumentException("The given parents are more numerous than the node's dimensions.");
       _parents = parents;
   }
   
   public void SetDimensions(ushort dimensions)
   { 
       _dimensions = dimensions;
       double[] old = new double[_weights.Length];
       for (int i = 0; i < _weights.Length; i++)
           old[i] = _weights[i];
       _weights = new double[dimensions];
       for (int i = 0; i < old.Length; i++)
           _weights[i] = old[i];
       for (int i = old.Length; i < _weights.Length; i++)
           _weights[i] = 1;
   }
   
   public void SetWeights(double[] weights)
   { 
       if (weights.Length != _dimensions) throw new ArgumentException("The new given weight array is greater in size than the previous, set a new dimension count beforehand.");
       _weights = weights;
   } 
   
   public void SetBias(double bias) => _bias = bias;
   
   public void SetActivation(ActivationType activation) => _activation = activation;
   
   public void SetParents(ushort[] parents)
   {
       if (_parents.Length > _dimensions) throw new ArgumentException("The given parents are more numerous than the node's dimensions.");
       _parents = parents;
   } 
   
   public ushort GetParentCount() => (ushort)_parents.Length;
   
   public ushort GetIdentifier() => _identifier;
   
   public ushort GetLayerIdentifier() => _layerIdentifier;
   
   public ushort GetDimensions() => _dimensions;
   
   public double[] GetWeights() => _weights;
   
   public double GetBias() => _bias;
   
   public ActivationType GetActivation() => _activation;
   
   public ushort[] GetParents() => _parents;
   
   public Node[] GetDirectParents(Network network)
   {
       Layer layer = network[_layerIdentifier - 1]; 
       Node[] parents = new Node[_parents.Length];
       for (int i = 0; i < _parents.Length; i++)
           parents[i] = layer[_parents[i]];
       return parents;
   }

   public Node[] GetDirectChildren(Network network)
   {
       Layer layer = network[_layerIdentifier + 1]; 
       List<Node> children = new List<Node>();
       for (int i = 0; i < layer.GetSize(); i++)
           if (layer[i].GetParents().Contains(_identifier)) children.Add(layer[i]);
       return children.ToArray();
   }
   
   public double Process(double[] input)
   {
       if (input.Length != _dimensions) throw new ArgumentException("The given input does not match the size expected by the node.");
       double result = WeightedSum(input);
       switch (_activation)
       {
           case ActivationType.Linear:
               return result;
           case ActivationType.Sigmoid:
               return ActivationFunction.Sigmoid(result);
           case ActivationType.Tanh:
               return ActivationFunction.TanH(result);
           case ActivationType.LeakyRElu:
               return ActivationFunction.LeakyReLU(result);
           case ActivationType.RElu:
               return ActivationFunction.ReLU(result);
           case ActivationType.AND:
               return ActivationFunction.AND(result);
           case ActivationType.NAND:
               return ActivationFunction.NAND(result);
           case ActivationType.OR:
               return ActivationFunction.OR(result);
           case ActivationType.NOR:
               return ActivationFunction.NOR(result);
           case ActivationType.EX:
               return ActivationFunction.EX(result);
           case ActivationType.NEX:
               return ActivationFunction.NEX(result);
           case ActivationType.Softmax:
               throw new Exception("Softmax should be handled at the layer level.");
           default:
               return 0;
       }
   }

   internal double WeightedSum(double[] input)
   {
       return DotProduct(input) + _bias;
   }

   private double DotProduct(double[] inputs)
   {
       int vectorSize = Vector<double>.Count;
       var sumVector = Vector<double>.Zero;
    
       for(int i=0; i <= inputs.Length - vectorSize; i += vectorSize)
       {
           var inputVector = new Vector<double>(inputs, i);
           var weightVector = new Vector<double>(_weights, i);
           sumVector += inputVector * weightVector;
       }
    
       double sum = Vector.Dot(sumVector, Vector<double>.One);
       
       for(int i = inputs.Length - (inputs.Length % vectorSize); i < inputs.Length; i++)
       {
           sum += inputs[i] * _weights[i];
       }
       return sum;
   }
   
   public override string ToString()
   {
       return $"{_dimensions};" + $"{string.Join(",", _weights.Select(w => w.ToString(CultureInfo.InvariantCulture)))};" + $"{_bias.ToString(CultureInfo.InvariantCulture)};" + $"{_activation};" + $"{string.Join(",", _parents)}" + "\n";
   }
}