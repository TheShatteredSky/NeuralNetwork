using System.Globalization;

namespace NeuralNetwork;

public static class NetworkUtilities
{
    private static Random _random = new Random();
    private static readonly object Lock = new object();
    
    internal static double RandomDouble(double minValue, double maxValue)
    {
        lock (Lock)
        {
            return _random.NextDouble() * (maxValue - minValue) + minValue;
        }
    }

    public static void NormalizeData(double[][] data)
    {
        if (data == null || data.Length == 0) return;
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
    
    public static void SaveToFile(string filePath, Network network) => File.WriteAllText(filePath, network.ToString());

    public static Network LoadFromFile(string filePath)
    {
        
        string[] lines = File.ReadAllLines(filePath);
        if (lines.Length == 0)
            throw new InvalidDataException("File is empty.");

        // Parse the first line (network metadata)
        string[] header = lines[0].Split(';');


        string name = header[0];
        string[] footer = lines.Last().Split(';');
        Network.LossFunction lossFunction = Enum.Parse<Network.LossFunction>(footer[0]);
        double baseLearningRate = double.Parse(footer[1], CultureInfo.InvariantCulture);
        ushort layerCount = ushort.Parse(header[1]);
        Network network = new Network(name);
        network.InstantiateBasics(layerCount - 2, lossFunction, baseLearningRate);
        int currentLine = 1; // Start reading layers after the header

        // Parse each layer
        for (int layerIdx = 0; layerIdx < layerCount; layerIdx++)
        {
            if (currentLine >= lines.Length)
                throw new InvalidDataException("Unexpected end of file while reading layers.");

            string[] layerHeader = lines[currentLine].Split(';');
            if (layerHeader.Length < 2)
                throw new InvalidDataException($"Invalid layer header at line {currentLine}.");

            Layer.LayerType layerType = Enum.Parse<Layer.LayerType>(layerHeader[0]);
            ushort layerSize = ushort.Parse(layerHeader[1]);
            currentLine++;

            Layer layer = new Layer((ushort)layerIdx, layerType);
            layer.InstantiateCustom(layerSize);
            network[layerIdx] = layer;

            // Parse nodes in this layer
            for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
            {
                if (currentLine >= lines.Length)
                    throw new InvalidDataException("Unexpected end of file while reading nodes.");

                string[] nodeData = lines[currentLine].Split(';');
                if (nodeData.Length < 5)
                    throw new InvalidDataException($"Invalid node data at line {currentLine}.");

                ushort dimensions = ushort.Parse(nodeData[0]);
                double[] weights = nodeData[1].Split(',')
                    .Select(w => double.Parse(w, CultureInfo.InvariantCulture))
                    .ToArray();
                double bias = double.Parse(nodeData[2], CultureInfo.InvariantCulture);
                Node.ActivationType activation = Enum.Parse<Node.ActivationType>(nodeData[3]);
                ushort[] parents = nodeData[4].Split(',')
                    .Select(p => ushort.Parse(p))
                    .ToArray();

                Node node = new Node(
                    (ushort)nodeIndex,
                    (ushort)layerIdx,
                    dimensions,
                    weights,
                    bias,
                    activation,
                    parents
                );

                layer.SetNodes(nodeIndex, node);
                currentLine++;
            }
        }

        return network;
    }
}