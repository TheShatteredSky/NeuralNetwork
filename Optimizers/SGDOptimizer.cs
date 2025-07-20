namespace NeuralNetwork.Optimizers;

public class SGDOptimizer : IOptimizer
{
    protected readonly Network Network;
    protected readonly LossType LossType;
    protected double LearningRate;
    
    public SGDOptimizer(Network network, LossType lossFunction, double baseLearningRate)
    {
        Network = network;
        LossType = lossFunction;
        LearningRate = baseLearningRate;
    }
    
    /// <summary>
    /// Optimizes the associated Network.
    /// </summary>
    /// <param name="data">The data to utilize.</param>
    /// <param name="totalEpochs">The number of epochs.</param>
    public virtual void Optimize(Dataset data, uint totalEpochs)
    {
        (double[][] unscaledInputs, double[][] unscaledOutputs) unscaled = Network.UnscaledData(data.GetInputs(), data.GetOutputs());
        double[][] inputs = unscaled.unscaledInputs;
        double[][] outputs = unscaled.unscaledOutputs;
        double[][][] weightGradientsForBatch = Utilities.InstantiateWeightArray(Network);
        double[][] biasGradientsForBatch = Utilities.InstantiateBiasArray(Network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(weightGradientsForBatch, biasGradientsForBatch, inputs, outputs);
            if (epoch % 100 == 0 && epoch > 0) LearningRate *= 0.9995;
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
        (double[][] unscaledInputs, double[][] unscaledOutputs) unscaled = Network.UnscaledData(data.GetInputs(), data.GetOutputs());
        double[][] inputs = unscaled.unscaledInputs;
        double[][] outputs = unscaled.unscaledOutputs;
        List<double> tracker = new List<double>();
        tracker.Add(Network.Loss(inputs, outputs, LossType));
        double[][][] weightGradientsForBatch = Utilities.InstantiateWeightArray(Network);
        double[][] biasGradientsForBatch = Utilities.InstantiateBiasArray(Network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(weightGradientsForBatch, biasGradientsForBatch, inputs, outputs);
            if (epoch % 100 == 0 && epoch > 0)
            {
                LearningRate *= 0.9995;
                tracker.Add(Network.Loss(inputs, outputs, LossType));
            }
        }
        return tracker.ToArray();
    }

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
                double[][][] weightGradientsPerLayer = Utilities.InstantiateWeightArray(Network);
                double[][] biasGradientsPerLayer = Utilities.InstantiateBiasArray(Network);
                NodeActivationRecord[][] forwardPassRecords = new NodeActivationRecord[Network.GetLayerCount()][];
                double[] layerInputs = new double[inputs[sampleIndex].Length];
                for (int i = 0; i < inputs[sampleIndex].Length; i++)
                    layerInputs[i] = inputs[sampleIndex][i];
                double[] predictions = ForwardPass(forwardPassRecords, layerInputs);
                double[] nextLayerDeltas = OutputLayerGradients(predictions, outputs, sampleIndex);
                BackPropagation(nextLayerDeltas, forwardPassRecords, weightGradientsPerLayer, biasGradientsPerLayer);
                AccumulateGradients(weightGradientsForBatch, biasGradientsForBatch, weightGradientsPerLayer, biasGradientsPerLayer, Network);
            }
            ApplyGradients(weightGradientsForBatch, biasGradientsForBatch, batchSize, Network, LearningRate);
        }
    }

    protected void BackPropagation(double[] nextLayerDeltas, NodeActivationRecord[][] forwardPassRecords, double[][][] weightGradientsPerLayer, double[][] biasGradientsPerLayer)
    {

        for (int layerIndex = Network.GetLayerCount() - 1; layerIndex >= 0; layerIndex--)
        {
            Layer currentLayer = Network[layerIndex];
            NodeActivationRecord[] layerActivationRecords = forwardPassRecords[layerIndex];
            double[] currentLayerDeltas = new double[currentLayer.GetSize()];

            for (int nodeIndex = 0; nodeIndex < currentLayer.GetSize(); nodeIndex++)
            {
                Node currentNode = currentLayer.GetNodes()[nodeIndex];
                NodeActivationRecord activationRecord = layerActivationRecords[nodeIndex];
                double delta;
                if (layerIndex == Network.GetLayerCount() - 1)
                    delta = nextLayerDeltas[nodeIndex];
                else
                {
                    delta = 0;
                    Layer downstreamLayer = Network[layerIndex + 1];
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
                    delta *= activationRecord.ActivationDerivative;
                }
                currentLayerDeltas[nodeIndex] = delta;

                for (int weightIndex = 0; weightIndex < currentNode.GetSize(); weightIndex++)
                    weightGradientsPerLayer[layerIndex][nodeIndex][weightIndex] +=
                        delta * activationRecord.InputValues[weightIndex];
                biasGradientsPerLayer[layerIndex][nodeIndex] += delta;
            }
            nextLayerDeltas = currentLayerDeltas;
        }
    }

    protected double[] OutputLayerGradients(double[] layerInputs, double[][] outputs, int sampleIndex)
    {
        double[] nextLayerDeltas = new double[layerInputs.Length];
        Layer outputLayer = Network[Network.GetLayerCount() - 1];
        for (int i = 0; i < outputLayer.GetSize(); i++)
        {
            double output = layerInputs[i];
            Node node = outputLayer.GetNodes()[i];

            switch (LossType)
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

    protected double[] ForwardPass(NodeActivationRecord[][] forwardPassRecords, double[] layerInputs)
    {
        for (int l = 0; l < Network.GetLayerCount(); l++)
        {
            Layer layer = Network[l];
            IReadOnlyList<Node> nodes = layer.GetNodes();
            NodeActivationRecord[] layerRecord = new NodeActivationRecord[layer.GetSize()];
            double[] layerOutputs = new double[layer.GetSize()];
            double[] weightedSums = layer.WeightedSums(layerInputs);
            double[]? softmaxOutputs = layer[0].GetActivation() == ActivationType.Softmax ? layer.SoftmaxOutputs(weightedSums) : null;
            for (int n = 0; n < layer.GetSize(); n++)
            {
                Node node = nodes[n];
                layerRecord[n] = NodeRecord(node, weightedSums, softmaxOutputs, n, layer, layerInputs);
                layerOutputs[n] = layerRecord[n].ActivationOutput;
            }
            forwardPassRecords[l] = layerRecord;
            layerInputs = layerOutputs;
        }
        return layerInputs;
    }

    protected NodeActivationRecord NodeRecord(Node node, double[] weightedSums, double[]? softmaxOutputs, int n, Layer layer, double[] inputs)
    {
        double activationOutput = 0;
        double activationDerivative = 0;
        switch (node.GetActivation())
        {
            case ActivationType.Softmax:
                activationOutput = softmaxOutputs![n];
                activationDerivative = 1;
                break;
            case ActivationType.Sigmoid:
                activationOutput = ActivationFunction.Sigmoid(weightedSums[n]);
                activationDerivative = activationOutput * (1 - activationOutput);
                break;
            case ActivationType.Linear:
                activationOutput = weightedSums[n];
                activationDerivative = 1;
                break;
            case ActivationType.RElu:
                activationOutput = ActivationFunction.ReLU(weightedSums[n]);
                activationDerivative = weightedSums[n] > 0 ? 1 : 0;
                break;
            case ActivationType.LeakyRElu:
                activationOutput = ActivationFunction.LeakyReLU(weightedSums[n]);
                activationDerivative = weightedSums[n] > 0 ? 1 : 0.01;
                break;
            case ActivationType.Tanh:
                activationOutput = ActivationFunction.TanH(weightedSums[n]);
                activationDerivative = 1 - activationOutput * activationOutput;
                break;
            //TODO: Other derivatives
        }
        return new NodeActivationRecord { InputValues = layer[n].NodeInputs(inputs), WeightedSum = weightedSums[n], ActivationOutput = activationOutput, ActivationDerivative = activationDerivative };
    }
    
    protected void AccumulateGradients(double[][][] weightAccumulator, double[][] biasAccumulator, double[][][] weights, double[][] biases, Network network)
    {
        for (int l = 0; l < network.GetLayerCount(); l++)
        {
            for (int n = 0; n < network[l].GetSize(); n++)
            {
                biasAccumulator[l][n] += biases[l][n];
                for (int w = 0; w < weights[l][n].Length; w++)
                    weightAccumulator[l][n][w] += weights[l][n][w];
            }
        }
    }
    
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
    
    //TODO: WeightedSum property might not be needed? To check.
    protected class NodeActivationRecord
    {
        public required double[] InputValues { get; init; } 
        public double WeightedSum { get; set; }
        public double ActivationOutput { get; set; }
        public double ActivationDerivative { get; init; }
    }
}