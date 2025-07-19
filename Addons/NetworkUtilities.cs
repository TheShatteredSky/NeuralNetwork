namespace NeuralNetwork.Addons;

/// <summary>
/// A collection of utility functions.
/// </summary>
public static class NetworkUtilities
{
    /// <summary>
    /// Normalizes the given data to a range of 0 to 1.
    /// Decrepit method, favorize using the scalers.
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
                if (range == 0)
                    data[i][j] = 0;
                else
                    data[i][j] = (data[i][j] - min[j]) / range;
            }
        }
    }

    /// <summary>
    /// The shifts, scales and de-shifts needed to UNSCALE data down to a specified range.
    /// </summary>
    /// <param name="data">The data for which the scales are needed.</param>
    /// <param name="tMin">The desired minimum of the range.</param>
    /// <param name="tMax">The desired maximum of the range.</param>
    /// <returns>The shift, scale and de-shift for each column of data.</returns>
    //TODO: I have no idea if a better way to unscale/scale data exists, this is the way I found but frankly seems inefficient.
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
    
    public static void SaveToFile(string filePath, Network network) => File.WriteAllText(filePath, network.ToString());

    public static Network LoadFromFile(string filePath)
    {
        string[] lines = File.ReadAllLines(filePath);
        if (lines.Length == 0) throw new InvalidDataException("File is empty.");
        string[] header = lines[0].Split(';');
        string name = header[0];
        ushort layerCount = ushort.Parse(header[1]);
        Network network = new Network();
        network.SetName(name);
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
            Layer layer = new Layer((ushort)layerIdx, layerType);
            layer.InstantiateCustom(layerSize);
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
                ushort[]? parents = nodeData[4] == "#" ? null : nodeData[4].Split(',').Select(ushort.Parse).ToArray();
                Node node = new Node((ushort)nodeIndex, (ushort)layerIdx, dimensions, weights, bias, activation, parents);
                layer[nodeIndex] = node;
                currentLine++;
            }
        }
        if (lines[currentLine + 1] != "#")
        {
            string[] inScales = lines[currentLine+1].Split(';'); 
            (double shift, double scale, double deshift)[] inputScales = new (double, double, double)[inScales.Length];
            for (int i = 0; i < inScales.Length; i++)
            {
                string scale = inScales[i];
                double[] nums = scale.Split(",").Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                inputScales[i] = (nums[0], nums[1], nums[2]);
            }
            network.SetInputScaling(inputScales);
        }
        if (lines[currentLine + 2] != "#")
        {
            string[] outScales = lines[currentLine + 2].Split(";");
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
    
    public static double[][][] InstantiateWeightArray(Network network)
    {
        double[][][] arr = new double[network.GetLayerCount()][][];
        for (int l = 0; l < network.GetLayerCount(); l++)
        {
            arr[l] = new double[network[l].GetSize()][];
            for (int n = 0; n < network[l].GetSize(); n++)
            {
                arr[l][n] = new double[network[l, n].GetDimensions()];
            }
        }
        return arr;
    }

    public static double[][] InstantiateBiasArray(Network network)
    {
        double[][] arr = new double[network.GetLayerCount()][];
        for (int l = 0; l < network.GetLayerCount(); l++)
            arr[l] = new double[network[l].GetSize()];
        return arr;
    }
    
    public static void ClearWeightArray(double[][][] networkWeights)
    {
        foreach (var layerWeights in networkWeights)
            foreach (var nodeWeights in layerWeights)
                for (int w = 0; w < nodeWeights.Length; w++)
                    nodeWeights[w] = 0;
    }

    public static void ClearBiasArray(double[][] networkBiases)
    {
        foreach (var layerBiases in networkBiases)
            for (int n = 0; n < layerBiases.Length; n++)
                layerBiases[n] = 0;
    }
    
    private static ThreadLocal<Random> _threadRandom = new (() => new Random(BitConverter.ToInt32(Guid.NewGuid().ToByteArray(), 0)));
    
    public static double NextDouble(double min, double max)
    {
        Random random = _threadRandom.Value!;
        return min + random.NextDouble() * (max - min);
    }

    public static void SetSeed(int seed)
    {
        _threadRandom = new  ThreadLocal<Random>(() => new Random(seed));
    }
    
    public static (double[], double)[][] StoreSettings(Network network)
    {
        return network.GetLayers().Select(layer => layer.GetNodes().Select(node => (node.GetWeights().ToArray(), node.GetBias())).ToArray()).ToArray();
    }

    public static void ExtractSettings((double[], double)[][] settings, Network network)
    {
        for (int l = 0; l < network.GetLayerCount(); l++)
        {
            for (int n = 0; n < network[l].GetSize(); n++)
            {
                network[l, n].SetBias(settings[l][n].Item2);
                for (int w = 0; w <network[l, n].GetDimensions(); w++)
                    network[l,n, w] = settings[l][n].Item1[w];
            }
        }
    }
    
    public static (double[][] inputs, double[][] outputs) GetData(string filePath)
    {
        string[] dataString = File.ReadAllLines(filePath);
        double[][] inputs = new double[dataString.Length][];
        double[][] outputs = new double[dataString.Length][];
        for (int i = 0; i < dataString.Length; i++)
        {
            string[] parts = dataString[i].Split(';');
            inputs[i] = parts[0].Split(",").Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
            outputs[i] = parts[1].Split(",").Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        }
        return (inputs, outputs);
    }
}