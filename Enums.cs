namespace NeuralNetwork;
  
  
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

public enum LayerType
{
    Input,
    Hidden,
    Output,
}

public enum LossType
{
    BinaryCrossEntropy, 
    CategoricalCrossEntropy,
    MSE
}

public enum OptimizerType
{
    SGD,
    Adam
}