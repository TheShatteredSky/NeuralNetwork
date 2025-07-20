namespace NeuralNetwork;
  
/// <summary>
/// Activation function types for Nodes.
/// </summary>
public enum ActivationType
{
    /// <summary>
    /// Linear activation, y = x.
    /// </summary>
    Linear,
    /// <summary>
    /// Rectified linear unit activation, y = Max(0, x)
    /// </summary>
    RElu,
    /// <summary>
    /// Leaky Rectified linear unit activation 0.01 * x for under 0
    /// </summary>
    LeakyRElu,
    /// <summary>
    /// Sigmoid activation, y = 1/(1 + e^-x)
    /// </summary>
    Sigmoid,
    /// <summary>
    /// Hyperbolic tangent activation, y = (1 - e^-x)/(1 + e^-x)
    /// </summary>
    Tanh,
    /// <summary>
    /// Softmax activation, split total sum of 1 across all outputs.
    /// </summary>
    Softmax,
    /// <summary>
    /// Quadratic imitating bitwise AND.
    /// </summary>
    AND,
    /// <summary>
    /// Quadratic imitating bitwise NAND.
    /// </summary>
    NAND,
    /// <summary>
    /// Quadratic imitating bitwise OR.
    /// </summary>
    OR,
    /// <summary>
    /// Quadratic imitating bitwise NOR.
    /// </summary>
    NOR,
    /// <summary>
    /// Quadratic imitating bitwise EX.
    /// </summary>
    EX,
    /// <summary>
    /// Quadratic imitating bitwise NEX.
    /// </summary>
    NEX
}

/// <summary>
/// Types of Layers.
/// </summary>
public enum LayerType
{
    /// <summary>
    /// First layer, takes in raw inputs, only one parent.
    /// </summary>
    Input,
    /// <summary>
    /// Any intermediate layer.
    /// </summary>
    Hidden,
    /// <summary>
    /// Outputs layer, produces the outputs.
    /// </summary>
    Output,
}

/// <summary>
/// Loss function types.
/// </summary>
public enum LossType
{
    /// <summary>
    /// For independent {0;1} outputs.
    /// </summary>
    BinaryCrossEntropy, 
    /// <summary>
    /// For Softmax.
    /// </summary>
    CategoricalCrossEntropy,
    /// <summary>
    /// For Linear stuff.
    /// </summary>
    MSE
}

/// <summary>
/// Optimizer types.
/// </summary>
public enum OptimizerType
{
    /// <summary>
    /// Base gradient descent (Stochastic Gradient Descent).
    /// </summary>
    SGD,
    /// <summary>
    /// Adam gradient descent (Adaptive Moment Estimation)
    /// </summary>
    Adam
}