namespace NeuralNetwork;

public class Neuron
{
    
   private readonly ushort _identifier;
   private readonly ushort _layerIdentifier;
   private ushort _dimensions;
   private double[] _weights;
   private double _bias;
   private ActivationType _activation;
   private ushort[] _parents;
   
   
   //Constructor for a neuron loaded from a save file.
   public Neuron(ushort identifier, ushort layerIdentifier, ushort dimensions, double[] weights, double bias, string? activation, ushort[] parents)
   {
       _identifier = identifier;
       _layerIdentifier = layerIdentifier;
       _dimensions = dimensions;
       switch (activation)
       {
           case "RElu":
               _activation = ActivationType.RElu;
               break;
           case "Sigmoid":
               _activation = ActivationType.Sigmoid;
               break;
           case "Tanh":
               _activation = ActivationType.Tanh;
               break;
           case "Linear":
               _activation = ActivationType.Linear;
               break;
           case "AND":
               _activation = ActivationType.AND;
               break;
           case "NAND":
               _activation = ActivationType.NAND;
               break;
           case "OR":
               _activation = ActivationType.OR;
               break;
           case "NOR":
               _activation = ActivationType.NOR;
               break;
           case "EX":
               _activation = ActivationType.EX;
               break;
           case "NEX":
               _activation = ActivationType.NEX;
               break;
           default:
               throw new ArgumentException($"Invalid activation type '{activation}' for neuron {_layerIdentifier}:{_identifier}");
       }
       _weights = weights;
       _bias = bias;
       _parents = parents;
   }
   
   //Newly created neuron constructor.
   public Neuron(ushort identifier, ushort layerIdentifier, ushort dimensions, string? activation, ushort[] parents)
   {
       _identifier = identifier;
       _layerIdentifier = layerIdentifier;
       _dimensions = dimensions;
       _weights = new double[dimensions];
       Random random = new Random();
       for (int i = 0; i < _weights.Length; i++)
           _weights[i] = random.NextDouble() - 0.5;
       _bias = random.NextDouble() - 0.5;
       switch (activation)
       {
           case "RElu":
               _activation = ActivationType.RElu;
               break;
           case "Sigmoid":
               _activation = ActivationType.Sigmoid;
               break;
           case "Tanh":
               _activation = ActivationType.Tanh;
               break;
           case "Linear":
               _activation = ActivationType.Linear;
               break;
           case "AND":
               _activation = ActivationType.AND;
               break;
           case "NAND":
               _activation = ActivationType.NAND;
               break;
           case "OR":
               _activation = ActivationType.OR;
               break;
           case "NOR":
               _activation = ActivationType.NOR;
               break;
           case "EX":
               _activation = ActivationType.EX;
               break;
           case "NEX":
               _activation = ActivationType.NEX;
               break;
           default:
               Console.WriteLine($"Neuron {_layerIdentifier};{_identifier} has invalid activation '{activation}'. Defaulting to Linear.");
               _activation = ActivationType.Linear;
               break;
       }
       _parents = parents;
   }
   
   public void SetDimensions(ushort dimensions) => _dimensions = dimensions;
   public void SetWeights(double[] weights) => _weights = weights;
   public void SetBias(double bias) => _bias = bias;
   public void SetActivation(ActivationType activation) => _activation = activation;
   public void SetParents(ushort[] parents) => _parents = parents;
   
   public ushort GetIdentifier() => _identifier;
   public ushort GetLayerIdentifier() => _layerIdentifier;
   public ushort GetDimensions() => _dimensions;
   public double[] GetWeights() => _weights;
   public double GetBias() => _bias;
   public ActivationType? GetActivation() => _activation;
   public ushort[] GetParents() => _parents;
   
   public enum ActivationType
   {
       RElu,
       Sigmoid,
       Tanh,
       Linear,
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
       double result = 0;
       for (int i = 0; i < _dimensions; i++)
           result += _weights[i] * input[i];
       result += _bias;
       switch (_activation)
       {
           case ActivationType.Linear:
               return result;
           case ActivationType.Sigmoid:
               return Functions.Sigmoid(result);
           case ActivationType.Tanh:
               return Math.Tanh(result);
           case ActivationType.RElu:
               return Math.Max(0, result);
           case ActivationType.AND:
               return result*result/4 + result/2 - 1;
           case ActivationType.NAND:
               return -result*result/4 - result/2 + 1;
           case ActivationType.OR:
               return -result*result/4 + result/2 + 1;
           case ActivationType.NOR:
               return result*result/4 - result/2 - 1;
           case ActivationType.EX:
               return -result*result/2 + 1;
           case ActivationType.NEX:
               return result*result/2 - 1;
       }
       return result;
   }
}
//A