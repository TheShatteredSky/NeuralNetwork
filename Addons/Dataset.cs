namespace NeuralNetwork.Addons;

/// <summary>
/// A Dataset instance.
/// </summary>
public class Dataset
{
    private string? _name;
    private double[][] _inputs;
    private double[][]? _outputs;

    /// <summary>
    /// Creates a new Dataset with  specified inputs.
    /// </summary>
    /// <param name="inputs">The inputs of the Dataset.</param>
    public Dataset(double[][] inputs)
    {
        _name = null;
        _inputs = inputs;
        _outputs = null;
    }
    
    /// <summary>
    /// Creates a new Dataset with a specified name and specified inputs.
    /// </summary>
    /// <param name="inputs">The inputs of the dataset.</param>
    /// <param name="name">The name of the dataset.</param>
    public Dataset(double[][] inputs, string name)
    {
        _name = name;
        _inputs = inputs;
        _outputs = null;
    }

    /// <summary>
    /// Creates a new Dataset with specified inputs and outputs.
    /// </summary>
    /// <param name="inputs">The inputs of the Dataset.</param>
    /// <param name="outputs">The outputs of the Dataset.</param>
    public Dataset(double[][] inputs, double[][] outputs)
    {
        _name = null;
        _inputs = inputs;
        _outputs = outputs;
    }
    
    /// <summary>
    /// Creates a new Dataset with a specified name and specified inputs and outputs.
    /// </summary>
    /// <param name="inputs">The inputs of the Dataset.</param>
    /// <param name="outputs">The outputs of the Dataset.</param>
    /// <param name="name">The name of the Dataset.</param>
    public Dataset(double[][] inputs, double[][] outputs, string name)
    {
        _name = name;
        _inputs = inputs;
        _outputs = outputs;
    }
    
    /// <summary>
    /// Fetches the name of this Dataset.
    /// </summary>
    /// <returns>This Dataset's name.</returns>
    public string? GetName() => _name;
    
    /// <summary>
    /// Fetches the inputs of this Dataset.
    /// </summary>
    /// <returns>This Dataset's inputs.</returns>
    /// <exception cref="Exception"></exception>
    public double[][] GetInputs() => _inputs;
    
    /// <summary>
    /// Fetches the outputs of this Dataset.
    /// </summary>
    /// <returns>This Dataset's outputs.</returns>
    /// <exception cref="Exception"></exception>
    public double[][] GetOutputs() => _outputs ?? throw new Exception("This dataset doesn't have predefined outputs.");

    /// <summary>
    /// Sets the name of this Dataset.
    /// </summary>
    /// <param name="name">The new name of this Dataset.</param>
    public void SetName(string name) => _name = name;

    /// <summary>
    /// Sets the inputs of this Dataset.
    /// </summary>
    /// <param name="inputs">The new inputs of this Dataset.</param>
    public void SetInputs(double[][] inputs) => _inputs = inputs;

    /// <summary>
    /// Sets the outputs of this dataset.
    /// </summary>
    /// <param name="outputs">The new outputs of this Dataset.</param>
    public void SetOutputs(double[][] outputs) => _outputs = outputs;
}