
namespace NeuralNetwork;

using System.Globalization;
using System.Numerics;

public class Node
{
    
   private readonly ushort _identifier;
   private readonly ushort _layerIdentifier;
   private ushort _dimensions;
   private double[] _weights;
   private double _bias;
   private ActivationType _activation;
   private ushort[] _parents;
   
   //Constructor for a node loaded from a save file.
   public Node(ushort identifier, ushort layerIdentifier, ushort dimensions, double[] weights, double bias, ActivationType activation, ushort[] parents)
   {
       _identifier = identifier;
       _layerIdentifier = layerIdentifier;
       _dimensions = dimensions;
       _activation = activation;
       _weights = weights;
       _bias = bias;
       _parents = parents;
   }
   
   //Newly created node constructor.
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
       _parents = parents;
   }
   
   public void SetDimensions(ushort dimensions) => _dimensions = dimensions;
   public void SetWeights(double[] weights) => _weights = weights;
   public void SetBias(double bias) => _bias = bias;
   public void SetActivation(ActivationType activation) => _activation = activation;
   public void SetParents(ushort[] parents) => _parents = parents;
   
   public ushort GetParentCount() => (ushort)(_parents?.Length ?? 0);
   public ushort GetIdentifier() => _identifier;
   public ushort GetLayerIdentifier() => _layerIdentifier;
   public ushort GetDimensions() => _dimensions;
   public double[] GetWeights() => _weights;
   public double GetBias() => _bias;
   public ActivationType GetActivation() => _activation;
   public ushort[] GetParents() => _parents;
   
   public enum ActivationType
   {
       RElu,
       LeakyRElu,
       Sigmoid,
       Tanh,
       Linear,
       Softmax,
       AND,
       NAND,
       OR,
       NOR,
       EX,
       NEX
   }
   
   //Base process function.
   public double Process(double[] input)
   {
       double result = WeightedSum(input);
       switch (_activation)
       {
           case ActivationType.Linear:
               return result;
           case ActivationType.Sigmoid:
               return Functions.Sigmoid(result);
           case ActivationType.Tanh:
               return Math.Tanh(result);
           case ActivationType.LeakyRElu:
               return result > 0 ? result : 0.01 * result;
           case ActivationType.RElu:
               return result > 0 ? result : 0;
           case ActivationType.AND:
               return Functions.And(result);
           case ActivationType.NAND:
               return Functions.Nand(result);
           case ActivationType.OR:
               return Functions.Or(result);
           case ActivationType.NOR:
               return Functions.Nor(result);
           case ActivationType.EX:
               return Functions.Ex(result);
           case ActivationType.NEX:
               return Functions.Nex(result);
           case ActivationType.Softmax:
               return result;
           default:
               return 0;
       }
   }

   internal double WeightedSum(double[] input)
   {
       return DotProduct(input) + _bias;
   }

   public double DotProduct(double[] inputs)
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
//