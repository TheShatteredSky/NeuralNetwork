namespace NeuralNetwork.Core;

/// <summary>
/// A modified Layer for spatial data.
/// </summary>
public class ConvolutionalLayer : Layer
{
    /// <summary>
    /// Creates a new Convolutional Layer.
    /// </summary>
    /// <param name="type">This Layer's type.</param>
    public ConvolutionalLayer(LayerType type) : base(type)
    {
        
    }
}