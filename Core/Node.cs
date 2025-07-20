namespace NeuralNetwork.Core;

using Addons; 
    
/// <summary>
/// A Node instance.
/// </summary>
public class Node
{
   private ushort _dimensions;
   private double[] _weights;
   private double _bias;
   private ActivationType _activation;
   private ushort[]? _parents;

   /// <summary>
   /// Indexer for this Node's parameters.
   /// </summary>
   /// <param name="parameterIndex">The index of the parameter, 0 to dimensions - 1 will return the specified weight, while dimensions will return the bias.</param>
   public double this[int parameterIndex]
   {
       get => parameterIndex < _dimensions ? _weights[parameterIndex] : _bias;
       set
       {
           if (parameterIndex < _dimensions) _weights[parameterIndex] = value;
           else _bias = value;
       }
   }
   
   /// <summary>
   /// Creates a new Node with predefined parameters.
   /// </summary>
   /// <param name="dimensions">The number of inputs the Node takes.</param>
   /// <param name="weights">The Node's weights.</param>
   /// <param name="bias">The Node's bias.</param>
   /// <param name="activation">The Node's activation function.</param>
   /// <param name="parents">The Node's parents.</param>
   /// <exception cref="ArgumentException"></exception>
   public Node(ushort dimensions, double[] weights, double bias, ActivationType activation, ushort[]? parents)
   {
       if (dimensions != weights.Length) throw new ArgumentException("The number of dimensions does not match the number of weights.");
       _dimensions = dimensions;
       _activation = activation;
       _weights = NetworkUtilities.CopyNonObjectArray(weights);
       _bias = bias;
       if (parents != null && parents.Length > dimensions) throw new ArgumentException("The given parents are more numerous than the Node's dimensions.");
       _parents = parents == null ? null : NetworkUtilities.CopyNonObjectArray(parents);
   }
   
   /// <summary>
   /// Creates a new Node.
   /// </summary>
   /// <param name="dimensions">The number of inputs the Node takes.</param>
   /// <param name="activation">The Node's activation function.</param>
   /// <param name="parents">The Node's parents.</param>
   /// <exception cref="ArgumentException"></exception>
   public Node(ushort dimensions, ActivationType activation, ushort[]? parents)
   {
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
       _activation = activation;
       if (parents != null && parents.Length > dimensions) throw new ArgumentException("The given parents are more numerous than the Node's dimensions.");
       _parents = parents == null ? null : NetworkUtilities.CopyNonObjectArray(parents);
   }
   
   /// <summary>
   /// Sets the amount of weights and parents this Node has.
   /// ⚠ This will completely wipe the Node's weights and set its parents to all.
   /// </summary>
   /// <param name="size">The new size of the Node.</param>
   public void SetSize(ushort size)
   {
       _weights = new double[size];
       _dimensions = size;
       _parents = null;
   }
   
   /// <summary>
   /// Sets the weights of this Node. The new weight array should be the same size as the previous. If you wish to change its size, use SetSize() first.
   /// </summary>
   /// <param name="weights">The new weights for the Node.</param>
   /// <exception cref="ArgumentException"></exception>
   public void SetWeights(double[] weights)
   { 
       if (weights.Length != _dimensions) throw new ArgumentException("The given weight array is greater in size than the current, set a new size beforehand.");
       _weights = NetworkUtilities.CopyNonObjectArray(weights);
   }
   
   /// <summary>
   /// Sets the bias of this Node.
   /// </summary>
   /// <param name="bias">The new bias for the Node.</param>
   public void SetBias(double bias) => _bias = bias;
   
   /// <summary>
   /// Sets the activation function of this Node.
   /// </summary>
   /// <param name="activation">The new activation function of the Node.</param>
   public void SetActivation(ActivationType activation) => _activation = activation;
   
   /// <summary>
   /// Sets the parents of this Node. The new parents array should be the same size as the previous. If you wish to change its size, use SetSize() first.
   /// </summary>
   /// <param name="parents">The new parents for the Node. Use null for all parents.</param>
   /// <exception cref="ArgumentException"></exception>
   public void SetParents(ushort[]? parents)
   {
       if (parents != null && parents.Length > _dimensions) throw new ArgumentException("The given parents are more numerous than the Node's dimensions.");
       _parents = parents == null ? null : NetworkUtilities.CopyNonObjectArray(parents);
   } 
   
   /// <summary>
   /// Fetches the number of parents this Node has.
   /// </summary>
   /// <returns>This Node's parent count.</returns>
   public ushort GetParentCount()
   {
       if (_parents == null) return _dimensions;
       return (ushort)_parents.Length;
   }
   
   /// <summary>
   /// Fetches the number of weights and parents this Node can have.
   /// </summary>
   /// <returns>This Node's dimensions count.</returns>
   public ushort GetSize() => _dimensions;
   
   /// <summary>
   /// Fetches the weights of this Node.
   /// </summary>
   /// <returns>This Node's weights.</returns>
   public double[] GetWeights() => _weights;
   
   /// <summary>
   /// Fetches the bias of this Node.
   /// </summary>
   /// <returns>This Node's bias.</returns>
   public double GetBias() => _bias;
   
   /// <summary>
   /// Fetches the activation function of this Node.
   /// </summary>
   /// <returns>This Node's activation function.</returns>
   public ActivationType GetActivation() => _activation;
   
   /// <summary>
   /// Fetches the parents of this Node.
   /// </summary>
   /// <returns>This Node's parents.</returns>
   public ushort[]? GetParents() => _parents;
   
   /// <summary>
   /// Filters the input array to only this Node's inputs.
   /// </summary>
   /// <param name="inputs">The original input array.</param>
   /// <returns>The filtered input array.</returns>
   internal double[] NodeInputs(double[] inputs)
   {
       double[] copy = NetworkUtilities.CopyNonObjectArray(inputs);
       return _parents == null ? copy : _parents.Select(x => copy[x]).ToArray();
   }
   
   /// <summary>
   /// Computes the outputs of this Node.
   /// </summary>
   /// <param name="input">The inputs for this Node.</param>
   /// <returns>The outputs of this Node.</returns>
   /// <exception cref="ArgumentException"></exception>
   /// <exception cref="Exception"></exception>
   public double Process(double[] input)
   {
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
               throw new Exception("Softmax should be handled at the Layer level.");
           default:
               return 0;
       }
   }

   /// <summary>
   /// Computes the weighted sum of this Node.
   /// </summary>
   /// <param name="input">The inputs for this Node.</param>
   /// <returns>The weighted sum of this Node.</returns>
   internal double WeightedSum(double[] input) => DotProduct(input) + _bias;

   /// <summary>
   /// Computes the dot product of this Node's weights and the inputs.
   /// </summary>
   /// <param name="inputs">The inputs for this Node.</param>
   /// <returns>The dot product of this Node.</returns>
   private double DotProduct(double[] inputs)
   {
       inputs = NodeInputs(inputs);
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
   
   /// <summary>
   /// The formatted string representing this Node.
   /// ⚠ This is a custom format, not JSON.
   /// </summary>
   /// <returns>A string representing this Node.</returns>
   public override string ToString() => $"{_dimensions};" + $"{string.Join(",", _weights.Select(w => w.ToString(CultureInfo.InvariantCulture)))};" + $"{_bias.ToString(CultureInfo.InvariantCulture)};" + $"{_activation};" + $"{(_parents == null ? "#" : string.Join(",", _parents))}" + "\n";
}