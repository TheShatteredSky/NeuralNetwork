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
        _inputs = new double[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            _inputs[i] = Utilities.CopyNonObjectArray(inputs[i]);
        _outputs = null;
    }

    /// <summary>
    /// Creates a new Dataset with a specified name and specified inputs.
    /// </summary>
    /// <param name="inputs">The inputs of the dataset.</param>
    /// <param name="name">The name of the dataset.</param>
    public Dataset(double[][] inputs, string? name)
    {
        _name = name;
        _inputs = new double[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            _inputs[i] = Utilities.CopyNonObjectArray(inputs[i]);
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
        _inputs = new double[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            _inputs[i] = Utilities.CopyNonObjectArray(inputs[i]);
        _outputs = new double[outputs.Length][];
        for (int i = 0; i < outputs.Length; i++)
            _outputs[i] = Utilities.CopyNonObjectArray(outputs[i]);
    }

    /// <summary>
    /// Creates a new Dataset with a specified name and specified inputs and outputs.
    /// </summary>
    /// <param name="inputs">The inputs of the Dataset.</param>
    /// <param name="outputs">The outputs of the Dataset.</param>
    /// <param name="name">The name of the Dataset.</param>
    public Dataset(double[][] inputs, double[][] outputs, string? name)
    {
        _name = name;
        _inputs = new double[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            _inputs[i] = Utilities.CopyNonObjectArray(inputs[i]);
        _outputs = new double[outputs.Length][];
        for (int i = 0; i < outputs.Length; i++)
            _outputs[i] = Utilities.CopyNonObjectArray(outputs[i]);
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
    public double[][]? GetOutputs() => _outputs;

    /// <summary>
    /// Sets the name of this Dataset.
    /// </summary>
    /// <param name="name">The new name of this Dataset.</param>
    public void SetName(string name) => _name = name;

    /// <summary>
    /// Sets the inputs of this Dataset.
    /// </summary>
    /// <param name="inputs">The new inputs of this Dataset.</param>
    public void SetInputs(double[][] inputs)
    {
        _inputs = new double[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
            _inputs[i] = Utilities.CopyNonObjectArray(inputs[i]);
    }

    /// <summary>
    /// Sets the outputs of this dataset.
    /// </summary>
    /// <param name="outputs">The new outputs of this Dataset.</param>
    public void SetOutputs(double[][] outputs)
    {
        _outputs = new double[outputs.Length][];
        for (int i = 0; i < outputs.Length; i++)
            _outputs[i] = Utilities.CopyNonObjectArray(outputs[i]);
    }


    /// <summary>
    /// Shuffles the data array, if the output array exists, it will be shuffled identically to the input array.
    /// </summary>
    public void Shuffle()
    {
        Random random = new Random();
        if (_outputs == null)
        {
            for (int i = 0; i < _inputs.Length; i++)
            {
                int index = random.Next(_inputs.Length);
                (_inputs[i], _inputs[index]) = (_inputs[index], _inputs[i]);
            }
        }
        else
        {
            for (int i = 0; i < _inputs.Length; i++)
            {
                int index = random.Next(_inputs.Length);
                (_inputs[i], _inputs[index]) = (_inputs[index], _inputs[i]);
                (_outputs[i], _outputs[index]) = (_outputs[index], _outputs[i]);
            }
        }
    } 
    
    /// <summary>
    /// Clones this Dataset.
    /// </summary>
    /// <returns>A copy of this Dataset.</returns>
    public Dataset Clone()
    {
        if (_outputs == null) return new Dataset(_inputs, _name);
        return new Dataset(_inputs, _outputs, _name);
    }

    /// <summary>
    /// Returns a string representation of this Dataset.
    /// </summary>
    /// <returns>A string representing this Dataset.</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine(_name ?? "null");
        if (_outputs == null)
        {
            for (int i = 0; i < _inputs.Length; i++)
                sb.AppendLine(string.Join(",", _inputs[i]));
        }
        else
        {
            for (int i = 0; i < _inputs.Length; i++)
            {
                sb.Append(string.Join(",", _inputs[i]) + ";");
                sb.AppendLine(string.Join(",", _outputs[i]));
            }
        }
        return sb.ToString();
    }
}