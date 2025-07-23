namespace NeuralNetwork.Optimizers;

/// <summary>
/// A SGDOptimizer instance.
/// </summary>
public class SGDOptimizer : IOptimizer
{
    private readonly Network _network;
    private readonly LossType _lossType;
    private double _learningRate;
    //null is none, false is L1, true is L2
    private bool? _regularization;

    /// <summary>
    /// Fetches the Network this SGDOptimizer is assigned too.
    /// </summary>
    /// <returns>The Network this instance is assigned too.</returns>
    public Network GetNetwork() => _network;
    
    /// <summary>
    /// Fetches this SGDOptimizer's utilized loss function.
    /// </summary>
    /// <returns>This SGDOptimizer's loss function.</returns>
    public LossType GetLossFunction() => _lossType;
    
    /// <summary>
    /// Fetches this SGDOptimizer's learning rate.
    /// </summary>
    /// <returns>This SGDOptimizer's current learning rate.</returns>
    public double GetLearningRate() => _learningRate;

    /// <summary>
    /// Sets this SGDOptimizer's learning rate.
    /// </summary>
    /// <param name="learningRate">The new learning rate.</param>
    public void SetLearningRate(double learningRate) => _learningRate = learningRate;
    
    /// <summary>
    /// Creates a new SGDOptimizer.
    /// </summary>
    /// <param name="network">The Network to optimize.</param>
    /// <param name="lossFunction">The loss function to utilize.</param>
    /// <param name="baseLearningRate">The base learning rate.</param>
    public SGDOptimizer(Network network, LossType lossFunction, double baseLearningRate)
    {
        _network = network;
        _lossType = lossFunction;
        _learningRate = baseLearningRate;
    }
    
    /// <summary>
    /// Optimizes the associated Network.
    /// </summary>
    /// <param name="data">The data to utilize.</param>
    /// <param name="totalEpochs">The number of epochs.</param>
    public virtual void Optimize(Dataset data, uint totalEpochs)
    {
        (double[][] unscaledInputs, double[][] unscaledOutputs) unscaled = _network.UnscaledData(data.GetInputs(), data.GetOutputs()!);
        double[][] inputs = unscaled.unscaledInputs;
        double[][] outputs = unscaled.unscaledOutputs;
        double[][][] weightGradientsForBatch = Utilities.InstantiateWeightArray(_network);
        double[][] biasGradientsForBatch = Utilities.InstantiateBiasArray(_network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(weightGradientsForBatch, biasGradientsForBatch, inputs, outputs);
            if (epoch % 100 == 0 && epoch > 0) _learningRate *= 0.9995;
        }
    }

    /// <summary>
    /// Optimizes the associated Networks and tracks the loss' evolution.
    /// </summary>
    /// <param name="data">The data to utilize.</param>
    /// <param name="totalEpochs">The number of epochs.</param>
    /// <returns>The evolution of the loss.</returns>
    public virtual double[] OptimizeTracked(Dataset data, uint totalEpochs)
    {
        (double[][] unscaledInputs, double[][] unscaledOutputs) unscaled = _network.UnscaledData(data.GetInputs(), data.GetOutputs()!);
        double[][] inputs = unscaled.unscaledInputs;
        double[][] outputs = unscaled.unscaledOutputs;
        List<double> tracker = [_network.Loss(data, _lossType)];
        double[][][] weightGradientsForBatch = Utilities.InstantiateWeightArray(_network);
        double[][] biasGradientsForBatch = Utilities.InstantiateBiasArray(_network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(weightGradientsForBatch, biasGradientsForBatch, inputs, outputs);
            if (epoch % 100 == 0 && epoch > 0)
            {
                _learningRate *= 0.9995;
                tracker.Add(_network.Loss(data, _lossType));
            }
        }
        return tracker.ToArray();
    }

    /// <summary>
    /// Executes an epoch.
    /// </summary>
    //Note: At no point should the data arrays be modified or returned, they should only be read.
    protected virtual void ExecuteEpoch(double[][][] weightGradientsForBatch, double[][] biasGradientsForBatch, double[][] inputs, double[][] outputs)
    {
        int sampleCount = outputs.Length;
        int batchSize = 64;
        int batches = (sampleCount + batchSize - 1) / batchSize;
        for (int batchIndex = 0; batchIndex < batches; batchIndex++)
        {
            Utilities.ClearWeightArray(weightGradientsForBatch);
            Utilities.ClearBiasArray(biasGradientsForBatch);
            for (int sampleIndex = batchIndex * batchSize; sampleIndex < sampleCount && sampleIndex < batchIndex * batchSize + batchSize; sampleIndex++)
            {
                double[][][] weightGradientsPerLayer = Utilities.InstantiateWeightArray(_network);
                double[][] biasGradientsPerLayer = Utilities.InstantiateBiasArray(_network);
                NodeRecord[][] forwardPassRecords = new NodeRecord[_network.GetLayerCount()][];
                double[] layerInputs = new double[inputs[sampleIndex].Length];
                for (int i = 0; i < inputs[sampleIndex].Length; i++)
                    layerInputs[i] = inputs[sampleIndex][i];
                double[] predictions = ForwardPass(forwardPassRecords, layerInputs);
                double[] nextLayerDeltas = OutputLayerGradients(predictions, outputs, sampleIndex);
                BackPropagation(nextLayerDeltas, forwardPassRecords, weightGradientsPerLayer, biasGradientsPerLayer);
                AccumulateGradients(weightGradientsForBatch, biasGradientsForBatch, weightGradientsPerLayer, biasGradientsPerLayer);
            }
            ApplyGradients(weightGradientsForBatch, biasGradientsForBatch, batchSize, _network, _learningRate);
        }
    }
    
    /// <summary>
    /// Passes forward through the Network to compute the NodeRecords.
    /// </summary>
    protected double[] ForwardPass(NodeRecord[][] forwardPassRecords, double[] layerInputs)
    {
        for (int l = 0; l < _network.GetLayerCount(); l++)
        {
            Layer layer = _network[l];
            IReadOnlyList<Node> nodes = layer.GetNodes();
            NodeRecord[] layerRecord = new NodeRecord[layer.GetSize()];
            double[] layerOutputs = new double[layer.GetSize()];
            double[] weightedSums = layer.WeightedSums(layerInputs);
            double[]? softmaxOutputs = layer[0].GetActivation() == ActivationType.Softmax ? layer.SoftmaxOutputs(weightedSums) : null;
            for (int n = 0; n < layer.GetSize(); n++)
            {
                Node node = nodes[n];
                layerRecord[n] = CreateNodeRecord(node, weightedSums[n], softmaxOutputs?[n], layerInputs);
                layerOutputs[n] = layerRecord[n].ActivationOutput;
            }
            forwardPassRecords[l] = layerRecord;
            layerInputs = layerOutputs;
        }
        return layerInputs;
    }

    /// <summary>
    /// Creates a specified NodeRecord.
    /// </summary>
    protected NodeRecord CreateNodeRecord(Node node, double weightedSum, double? softmaxOutput, double[] inputs)
    {
        double activationOutput = 0;
        double activationDerivative = 0;
        switch (node.GetActivation())
        {
            case ActivationType.Softmax:
                activationOutput = (double)softmaxOutput!;
                activationDerivative = 1;
                break;
            case ActivationType.Sigmoid:
                activationOutput = ActivationFunction.Sigmoid(weightedSum);
                activationDerivative = activationOutput * (1 - activationOutput);
                break;
            case ActivationType.Linear:
                activationOutput = weightedSum;
                activationDerivative = 1;
                break;
            case ActivationType.RElu:
                activationOutput = ActivationFunction.ReLU(weightedSum);
                activationDerivative = weightedSum > 0 ? 1 : 0;
                break;
            case ActivationType.LeakyRElu:
                activationOutput = ActivationFunction.LeakyReLU(weightedSum);
                activationDerivative = weightedSum > 0 ? 1 : 0.01;
                break;
            case ActivationType.Tanh:
                activationOutput = ActivationFunction.TanH(weightedSum);
                activationDerivative = 1 - activationOutput * activationOutput;
                break;
            case ActivationType.AND:
                activationOutput = ActivationFunction.AND(weightedSum);
                activationDerivative = weightedSum - 0.5;
                break;
            case ActivationType.NAND:
                activationOutput = ActivationFunction.NAND(weightedSum);
                activationDerivative = -weightedSum + 0.5;
                break;
            case ActivationType.OR:
                activationOutput = ActivationFunction.OR(weightedSum);
                activationDerivative = -weightedSum + 1.5;
                break;
            case ActivationType.NOR:
                activationOutput = ActivationFunction.NOR(weightedSum);
                activationDerivative = weightedSum - 1.5;
                break;
            case ActivationType.EX:
                activationOutput = ActivationFunction.EX(weightedSum);
                activationDerivative = -2 * weightedSum + 2;
                break;
            case ActivationType.NEX:
                activationOutput = ActivationFunction.NEX(weightedSum);
                activationDerivative = 2 * weightedSum - 2;
                break;
            
        }
        return new NodeRecord {InputValues = node.NodeInputs(inputs), ActivationOutput = activationOutput, ActivationDerivative = activationDerivative};
    }
    
    /// <summary>
    /// Computes the gradients for the output Layer.
    /// </summary>
    protected double[] OutputLayerGradients(double[] layerInputs, double[][] outputs, int sampleIndex)
    {
        double[] nextLayerDeltas = new double[layerInputs.Length];
        Layer outputLayer = _network[_network.GetLayerCount() - 1];
        for (int i = 0; i < outputLayer.GetSize(); i++)
        {
            double output = layerInputs[i];
            Node node = outputLayer.GetNodes()[i];

            switch (_lossType)
            {
                case LossType.BinaryCrossEntropy:
                    if (node.GetActivation() == ActivationType.Sigmoid) nextLayerDeltas[i] = output - outputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
                case LossType.CategoricalCrossEntropy:
                    if (node.GetActivation() == ActivationType.Softmax) nextLayerDeltas[i] = output - outputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
                case LossType.MSE:
                    if (node.GetActivation() == ActivationType.Linear) nextLayerDeltas[i] = 2 * (output - outputs[sampleIndex][i]);
                    else throw new NotImplementedException();
                    break;
            }
        }
        return nextLayerDeltas;
    }
    
    /// <summary>
    /// Backpropagates through the Network to compute the gradients.
    /// </summary>
    protected void BackPropagation(double[] nextLayerDeltas, NodeRecord[][] forwardPassRecords, double[][][] weightGradientsPerLayer, double[][] biasGradientsPerLayer)
    {

        for (int layerIndex = _network.GetLayerCount() - 1; layerIndex >= 0; layerIndex--)
        {
            Layer currentLayer = _network[layerIndex];
            NodeRecord[] layerActivationRecords = forwardPassRecords[layerIndex];
            double[] currentLayerDeltas = new double[currentLayer.GetSize()];

            for (int nodeIndex = 0; nodeIndex < currentLayer.GetSize(); nodeIndex++)
            {
                Node currentNode = currentLayer.GetNodes()[nodeIndex];
                NodeRecord record = layerActivationRecords[nodeIndex];
                double delta;
                if (layerIndex == _network.GetLayerCount() - 1)
                    delta = nextLayerDeltas[nodeIndex];
                else
                {
                    delta = 0;
                    Layer downstreamLayer = _network[layerIndex + 1];
                    for (int n = 0; n < downstreamLayer.GetSize(); n++)
                    {
                        Node downstreamNode = downstreamLayer[n];
                        IReadOnlyList<ushort>? parents = downstreamNode.GetParents();
                        if (parents == null) delta += downstreamNode.GetWeights()[nodeIndex] * nextLayerDeltas[n];
                        else
                        {
                            for (int parentIndex = 0; parentIndex < parents.Count; parentIndex++)
                                if (parents[parentIndex] == nodeIndex)
                                    delta += downstreamNode.GetWeights()[parentIndex] * nextLayerDeltas[n];
                        }
                    }
                    delta *= record.ActivationDerivative;
                }
                currentLayerDeltas[nodeIndex] = delta;

                for (int weightIndex = 0; weightIndex < currentNode.GetSize(); weightIndex++)
                    weightGradientsPerLayer[layerIndex][nodeIndex][weightIndex] +=
                        delta * record.InputValues[weightIndex];
                biasGradientsPerLayer[layerIndex][nodeIndex] += delta;
            }
            nextLayerDeltas = currentLayerDeltas;
        }
    }
    
    /// <summary>
    /// Accumulates the computed gradients into an accumulator.
    /// </summary>
    protected void AccumulateGradients(double[][][] weightAccumulator, double[][] biasAccumulator, double[][][] weights, double[][] biases)
    {
        for (int l = 0; l < biasAccumulator.Length; l++)
        {
            for (int n = 0; n < biasAccumulator[l].Length; n++)
            {
                biasAccumulator[l][n] += biases[l][n];
                for (int w = 0; w < weights[l][n].Length; w++)
                    weightAccumulator[l][n][w] += weights[l][n][w];
            }
        }
    }
    
    /// <summary>
    /// Applies the specified gradients to the assigned Network.
    /// </summary>
    protected virtual void ApplyGradients(double[][][] totalWeightGradients, double[][] totalBiasGradients, int sampleCount, Network network, double learningRate)
    {
        for (int layerIndex = 0; layerIndex < network.GetLayerCount(); layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < network[layerIndex].GetSize(); nodeIndex++)
            {
                for (int weightIndex = 0; weightIndex < network[layerIndex, nodeIndex].GetSize(); weightIndex++)
                    network[layerIndex, nodeIndex, weightIndex] -= learningRate * totalWeightGradients[layerIndex][nodeIndex][weightIndex] / sampleCount;
                network[layerIndex, nodeIndex, network[layerIndex, nodeIndex].GetSize()] -= learningRate * totalBiasGradients[layerIndex][nodeIndex] / sampleCount;
            }
        }
    }
    
    /// <summary>
    /// Instance of a record for a Node.
    /// </summary>
    protected class NodeRecord
    {
        /// <summary>
        /// The inputs of the Node.
        /// </summary>
        public required double[] InputValues { get; init; } 
        /// <summary>
        /// The output of the Node's activation.
        /// </summary>
        public required double ActivationOutput { get; init; }
        /// <summary>
        /// The output of the Node's derivatived activation.
        /// </summary>
        public required double ActivationDerivative { get; init; }
    }
}