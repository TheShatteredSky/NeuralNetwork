namespace NeuralNetwork.Addons;

/// <summary>
/// A collection of utility functions.
/// </summary>
public static class Utilities
{
    private static ThreadLocal<Random> _threadRandom = new (() => new Random(BitConverter.ToInt32(Guid.NewGuid().ToByteArray(), 0)));
    
    /// <summary>
    /// Normalizes the given data to a range of 0 to 1.
    /// Decrepit method, favorize using the built-in scalers.
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    public static void NormalizeData(double[][] data)
    {
        if (data.Length == 0) return;
        int rows = data.Length;
        int cols = data[0].Length;
        double[] min = new double[cols];
        double[] max = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            min[j] = double.MaxValue;
            max[j] = double.MinValue;
            for (int i = 0; i < rows; i++)
            {
                if (data[i][j] < min[j]) min[j] = data[i][j];
                if (data[i][j] > max[j]) max[j] = data[i][j];
            }
        }
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double range = max[j] - min[j];
                if (range == 0) data[i][j] = 0;
                else data[i][j] = (data[i][j] - min[j]) / range;
            }
        }
    }

    /// <summary>
    /// The shifts, scales and de-shifts needed to UNSCALE data into a specified inner range.
    /// </summary>
    /// <param name="data">The data for which the scales are needed.</param>
    /// <param name="tMin">The desired minimum of the range.</param>
    /// <param name="tMax">The desired maximum of the range.</param>
    /// <returns>The shift, scale and de-shift for each column of data.</returns>
    //TODO: I have no idea if a better way to unscale/scale data exists, this is the way I found but it frankly seems inefficient.
    public static (double shift, double scale, double deshift)[] GetScales(double[][] data, double tMin, double tMax)
    {
        int rows = data.Length;
        int cols = data[0].Length;
        (double shift, double scale, double deshift)[] scales = new (double, double, double)[cols];
        for (int j = 0; j < cols; j++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            for (int i = 0; i < rows; i++)
            {
                if (data[i][j] < min) min = data[i][j];
                if (data[i][j] > max) max = data[i][j];
            }
            scales[j].shift = -(min + (max - min) / 2 );
            scales[j].scale = (tMax - tMin) / (max - min);
            scales[j].deshift = tMin + (tMax - tMin) / 2;
        }
        return scales;
    }
    
    /// <summary>
    /// Saves the specified Network to the specified save file.
    /// </summary>
    /// <param name="filePath">The path for the save file.</param>
    /// <param name="network">The Network to save.</param>
    public static void SaveNetwork(string filePath, Network network) => File.WriteAllTextAsync(filePath, network.ToString());

    /// <summary>
    /// Saves the specified Dataset to the specified save file.
    /// </summary>
    /// <param name="filePath">The path for the save file.</param>
    /// <param name="dataset">The Dataset to save.</param>
    public static void SaveDataset(string filePath, Dataset dataset) => File.AppendAllTextAsync(filePath, dataset.ToString());

    /// <summary>
    /// Loads a Network from its save file.
    /// </summary>
    /// <param name="filePath">The path of the save file.</param>
    /// <returns>The loaded Network.</returns>
    /// <exception cref="InvalidDataException"></exception>
    public static Network LoadNetwork(string filePath)
    {
        string[] lines = File.ReadAllLines(filePath);
        if (lines.Length == 0) throw new InvalidDataException("File is empty.");
        string[] header = lines[0].Split(';');
        string name = header[0];
        ushort layerCount = ushort.Parse(header[1]);
        Network network = new Network();
        if (name != "null") network.SetName(name);
        network.Instantiate(layerCount - 2);
        int currentLine = 1;
        for (int layerIdx = 0; layerIdx < layerCount; layerIdx++)
        {
            if (currentLine >= lines.Length) throw new InvalidDataException("Unexpected end of file while reading layers.");
            string[] layerHeader = lines[currentLine].Split(';');
            if (layerHeader.Length < 2) throw new InvalidDataException($"Invalid layer header at line {currentLine}.");
            LayerType layerType = Enum.Parse<LayerType>(layerHeader[0]);
            ushort layerSize = ushort.Parse(layerHeader[1]);
            currentLine++;
            Layer layer = new Layer(layerType);
            layer.Instantiate(layerSize);
            network[layerIdx] = layer;
            for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
            {
                if (currentLine >= lines.Length)
                    throw new InvalidDataException("Unexpected end of file while reading nodes.");
                string[] nodeData = lines[currentLine].Split(';');
                if (nodeData.Length < 5)
                    throw new InvalidDataException($"Invalid node data at line {currentLine}.");
                ushort dimensions = ushort.Parse(nodeData[0]);
                double[] weights = nodeData[1].Split(',').Select(w => double.Parse(w, CultureInfo.InvariantCulture)).ToArray();
                double bias = double.Parse(nodeData[2], CultureInfo.InvariantCulture);
                ActivationType activation = Enum.Parse<ActivationType>(nodeData[3]);
                ushort[]? parents = nodeData[4] == "null" ? null : nodeData[4].Split(',').Select(ushort.Parse).ToArray();
                Node node = new Node(dimensions, weights, bias, activation, parents);
                layer[nodeIndex] = node;
                currentLine++;
            }
        }
        if (lines[currentLine] != "null")
        {
            string[] inScales = lines[currentLine].Split(';'); 
            (double shift, double scale, double deshift)[] inputScales = new (double, double, double)[inScales.Length];
            for (int i = 0; i < inScales.Length; i++)
            {
                string scale = inScales[i];
                double[] nums = scale.Split(",").Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                inputScales[i] = (nums[0], nums[1], nums[2]);
            }
            network.SetInputScaling(inputScales);
        }
        if (lines[currentLine + 1] != "null")
        {
            string[] outScales = lines[currentLine + 1].Split(";");
            (double shift, double scale, double deshift)[] outputScales = new (double, double, double)[outScales.Length];
            for (int i = 0; i < outScales.Length; i++)
            {
                string scale = outScales[i];
                double[] nums = scale.Split(",").Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                outputScales[i] = (nums[0], nums[1], nums[2]);
            }
            network.SetOutputScaling(outputScales);
        }
        return network;
    }
    
    /// <summary>
    /// Loads a Dataset from its save file.
    /// </summary>
    /// <param name="filePath">The path of the save file.</param>
    /// <returns>The loaded Dataset.</returns>
    /// <exception cref="InvalidDataException"></exception>
    public static Dataset LoadDataset(string filePath)
    {
        string[] dataString = File.ReadAllLines(filePath);
        double[][] inputs = new double[dataString.Length - 1][];
        double[][] outputs = new double[dataString.Length - 1][];
        for (int i = 0; i < dataString.Length - 1; i++)
        {
            string[] parts = dataString[i + 1].Split(';');
            inputs[i] = parts[0].Split(",").Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
            outputs[i] = parts[1].Split(",").Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        }
        return new Dataset(inputs, outputs, dataString[0] == "null" ? null : dataString[0]);
    }
    
    /// <summary>
    /// Instantiates an array shaped like the weights of the specified Network.
    /// </summary>
    /// <param name="network">The Network to create the array of.</param>
    /// <returns>A weight-shaped array.</returns>
    public static double[][][] InstantiateWeightArray(Network network)
    {
        double[][][] arr = new double[network.GetLayerCount()][][];
        for (int l = 0; l < network.GetLayerCount(); l++)
        {
            arr[l] = new double[network[l].GetSize()][];
            for (int n = 0; n < network[l].GetSize(); n++)
            {
                arr[l][n] = new double[network[l, n].GetSize()];
            }
        }
        return arr;
    }

    /// <summary>
    /// Instantiates an array shaped like the biases of the specified Network.
    /// </summary>
    /// <param name="network">The Network to create the array of.</param>
    /// <returns>A bias-shaped array.</returns>
    public static double[][] InstantiateBiasArray(Network network)
    {
        double[][] arr = new double[network.GetLayerCount()][];
        for (int l = 0; l < network.GetLayerCount(); l++)
            arr[l] = new double[network[l].GetSize()];
        return arr;
    }
    
    /// <summary>
    /// Clears the specified weight array.
    /// </summary>
    /// <param name="networkWeights">The weight array to clear.</param>
    public static void ClearWeightArray(double[][][] networkWeights)
    {
        foreach (var layerWeights in networkWeights)
            foreach (var nodeWeights in layerWeights)
                for (int w = 0; w < nodeWeights.Length; w++)
                    nodeWeights[w] = 0;
    }

    /// <summary>
    /// Clears the specified bias array.
    /// </summary>
    /// <param name="networkBiases">The bias array to clear.</param>
    public static void ClearBiasArray(double[][] networkBiases)
    {
        foreach (var layerBiases in networkBiases)
            for (int n = 0; n < layerBiases.Length; n++)
                layerBiases[n] = 0;
    }
    
    /// <summary>
    /// Computes a semi-random double within the desired range.
    /// </summary>
    /// <param name="min">The minimum of the desired range.</param>
    /// <param name="max">The maximum of the desired range.</param>
    /// <returns></returns>
    public static double RandomDouble(double min, double max)
    {
        Random random = _threadRandom.Value!;
        return min + random.NextDouble() * (max - min);
    }

    /// <summary>
    /// Sets the seed of the Random object.
    /// </summary>
    /// <param name="seed">The new seed.</param>
    public static void SetSeed(int seed)
    {
        _threadRandom = new  ThreadLocal<Random>(() => new Random(seed));
    }
    
    /// <summary>
    /// Stores the parameters of the specified Network.
    /// </summary>
    /// <param name="network">The Network of which to store the parameters.</param>
    /// <returns>The stored parameters of the specified Network.</returns>
    public static (double[], double)[][] StoreParameters(Network network)
    {
        return network.GetLayers().Select(layer => layer.GetNodes().Select(node => (node.GetWeights().ToArray(), node.GetBias())).ToArray()).ToArray();
    }

    /// <summary>
    /// Extracts the parameters into the specified Network.
    /// </summary>
    /// <param name="settings">The parameters to extract.</param>
    /// <param name="network">The Network to apply those parameters onto.</param>
    public static void ExtractParameters((double[], double)[][] settings, Network network)
    {
        for (int l = 0; l < network.GetLayerCount(); l++)
        {
            for (int n = 0; n < network[l].GetSize(); n++)
            {
                network[l, n].SetBias(settings[l][n].Item2);
                for (int w = 0; w <network[l, n].GetSize(); w++)
                    network[l,n, w] = settings[l][n].Item1[w];
            }
        }
    }

    /// <summary>
    /// Creates a copy of primitive value arrays.
    /// ⚠ Do not use this on reference-type arrays.
    /// </summary>
    /// <param name="original">The array to copy.</param>
    /// <typeparam name="T"></typeparam>
    /// <returns>The copied array.</returns>
    //Frankly, I don't understand why there's no built-in method for this. Do not use this on Object arrays.
    internal static T[] CopyNonObjectArray<T>(T[] original)
    {
        T[] result = new T[original.Length];
        for (int i = 0; i < original.Length; i++)
            result[i] = original[i];
        return result;
    }

    internal static IReadOnlyList<T> ConvertToReadOnlyList<T>(T[] original)
    {
        IReadOnlyList<T> result = new List<T>(original);
        return result;
    }
}