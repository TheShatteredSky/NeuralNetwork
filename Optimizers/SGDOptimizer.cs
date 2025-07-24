namespace NeuralNetwork.Optimizers;

/// <summary>
/// A SGDOptimizer instance.
/// </summary>
public class SGDOptimizer : IOptimizer
{
    protected readonly Network Network;
    protected readonly LossType LossType;
    protected double LearningRate;
    protected double LearningRateDecay;
    //null is none, true is L1, false is L2
    protected bool? L1Regularization;
    protected double RegularizationFactor;
    
    
    /// <summary>
    /// Creates a new SGDOptimizer.
    /// </summary>
    /// <param name="network">The Network to optimize.</param>
    /// <param name="lossFunction">The loss function to utilize.</param>
    /// <param name="baseLearningRate">The base learning rate.</param>
    public SGDOptimizer(Network network, LossType lossFunction, double baseLearningRate)
    {
        Network = network;
        LossType = lossFunction;
        LearningRate = baseLearningRate;
        RegularizationFactor = 0.000315;
        L1Regularization = null;
        LearningRateDecay = 0.9995;
    }

    public bool? GetRegularization() => L1Regularization;
    public void SetRegularization(bool? reg) => L1Regularization = reg;
    
    /// <summary>
    /// Optimizes the associated Network.
    /// </summary>
    /// <param name="data">The data to utilize.</param>
    /// <param name="totalEpochs">The number of epochs.</param>
    public virtual void Optimize(Dataset data, uint totalEpochs)
    {
        (double[][] unscaledInputs, double[][] unscaledOutputs) unscaled = Network.UnscaledData(data.GetInputs(), data.GetOutputs()!);
        double[][] inputs = unscaled.unscaledInputs;
        double[][] outputs = unscaled.unscaledOutputs;
        double[][][] weightGradientsForBatch = Utilities.InstantiateWeightArray(Network);
        double[][] biasGradientsForBatch = Utilities.InstantiateBiasArray(Network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(weightGradientsForBatch, biasGradientsForBatch, inputs, outputs);
            if (epoch % 100 == 0 && epoch > 0) LearningRate *= LearningRateDecay;
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
        (double[][] unscaledInputs, double[][] unscaledOutputs) unscaled = Network.UnscaledData(data.GetInputs(), data.GetOutputs()!);
        double[][] inputs = unscaled.unscaledInputs;
        double[][] outputs = unscaled.unscaledOutputs;
        List<double> tracker = [Network.Loss(data, LossType)];
        double[][][] weightGradientsForBatch = Utilities.InstantiateWeightArray(Network);
        double[][] biasGradientsForBatch = Utilities.InstantiateBiasArray(Network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(weightGradientsForBatch, biasGradientsForBatch, inputs, outputs);
            if (epoch % 100 == 0 && epoch > 0)
            {
                LearningRate *= LearningRateDecay;
                tracker.Add(Network.Loss(data, LossType));
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
                double[][][] weightGradientsPerLayer = Utilities.InstantiateWeightArray(Network);
                double[][] biasGradientsPerLayer = Utilities.InstantiateBiasArray(Network);
                NodeRecord[][] forwardPassRecords = new NodeRecord[Network.GetLayerCount()][];
                double[] layerInputs = new double[inputs[sampleIndex].Length];
                for (int i = 0; i < inputs[sampleIndex].Length; i++)
                    layerInputs[i] = inputs[sampleIndex][i];
                double[] predictions = ForwardPass(forwardPassRecords, layerInputs);
                double[] nextLayerDeltas = OutputLayerGradients(predictions, outputs, sampleIndex);
                BackPropagation(nextLayerDeltas, forwardPassRecords, weightGradientsPerLayer, biasGradientsPerLayer);
                AccumulateGradients(weightGradientsForBatch, biasGradientsForBatch, weightGradientsPerLayer, biasGradientsPerLayer);
            }
            ScaleGradients(weightGradientsForBatch, biasGradientsForBatch, batchSize);
            ApplyRegularizationToGradients(weightGradientsForBatch);
            ApplyGradients(weightGradientsForBatch, biasGradientsForBatch, Network, LearningRate);
        }
    }
    
    /// <summary>
    /// Divides all gradients by a constant (equal to the batch size).
    /// </summary>
    protected void ScaleGradients(double[][][] weightGradientsForBatch, double[][] biasGradientsForBatch, int size)
    {
        for (int layerIndex = 0; layerIndex < weightGradientsForBatch.Length; layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < weightGradientsForBatch[layerIndex].Length; nodeIndex++)
            {
                for (int weightIndex = 0; weightIndex < weightGradientsForBatch[layerIndex][nodeIndex].Length; weightIndex++)
                    weightGradientsForBatch[layerIndex][nodeIndex][weightIndex] /= size;
                biasGradientsForBatch[layerIndex][nodeIndex] /= size;
            }
        }
    }
    
    /// <summary>
    /// Passes forward through the Network to compute the NodeRecords.
    /// </summary>
    protected double[] ForwardPass(NodeRecord[][] forwardPassRecords, double[] layerInputs)
    {
        for (int l = 0; l < Network.GetLayerCount(); l++)
        {
            Layer layer = Network[l];
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
    
    /// <summary>
    /// Backpropagates through the Network to compute the gradients.
    /// </summary>
    protected void BackPropagation(double[] nextLayerDeltas, NodeRecord[][] forwardPassRecords, double[][][] weightGradientsPerLayer, double[][] biasGradientsPerLayer)
    {

        for (int layerIndex = Network.GetLayerCount() - 1; layerIndex >= 0; layerIndex--)
        {
            Layer currentLayer = Network[layerIndex];
            NodeRecord[] layerActivationRecords = forwardPassRecords[layerIndex];
            double[] currentLayerDeltas = new double[currentLayer.GetSize()];

            for (int nodeIndex = 0; nodeIndex < currentLayer.GetSize(); nodeIndex++)
            {
                Node currentNode = currentLayer.GetNodes()[nodeIndex];
                NodeRecord record = layerActivationRecords[nodeIndex];
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
    protected virtual void ApplyGradients(double[][][] totalWeightGradients, double[][] totalBiasGradients, Network network, double learningRate)
    {
        for (int layerIndex = 0; layerIndex < network.GetLayerCount(); layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < network[layerIndex].GetSize(); nodeIndex++)
            {
                for (int weightIndex = 0; weightIndex < network[layerIndex, nodeIndex].GetSize(); weightIndex++)
                    network[layerIndex, nodeIndex, weightIndex] -= learningRate * totalWeightGradients[layerIndex][nodeIndex][weightIndex];
                network[layerIndex, nodeIndex, network[layerIndex, nodeIndex].GetSize()] -= learningRate * totalBiasGradients[layerIndex][nodeIndex];
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

    /// <summary>
    /// Computers the additional loss off regularization.
    /// </summary>
    /// <param name="weights">The weights of the Network.</param>
    /// <returns>The additional regularization loss.</returns>
    public static double RegularizationLoss(double[][][] weights, bool l1Regularization, double regularizationFactor)
    {
        double result = 0;
        if (l1Regularization)
        {
            foreach (double[][] layerWeights in weights)
                foreach (double[] nodeWeights in layerWeights)
                    foreach (double weight in nodeWeights)
                        result += Math.Abs(weight);
            return regularizationFactor * result;
        }
        foreach (double[][] layerWeights in weights)
            foreach (double[] nodeWeights in layerWeights)
                foreach (double weight in nodeWeights)
                    result += weight * weight;
        return regularizationFactor / 2 * result;
    }
    
    /// <summary>
    /// Applies the regularization penalties to the Network's computed gradients.
    /// </summary>
    /// <param name="weightGradients">The previously computer gradients.</param>
    protected void ApplyRegularizationToGradients(double[][][] weightGradients)
    {
        if (L1Regularization == null) return;
        if ((bool)L1Regularization)
        {
            for (int l = 0; l < weightGradients.Length; l++) 
                for (int n = 0; n < weightGradients[l].Length; n++)
                    for (int w = 0; w < weightGradients[l][n].Length; w++)
                        weightGradients[l][n][w] += RegularizationFactor * Math.Sign(Network[l][n][w]);
        }
        else
        {
            for (int l = 0; l < weightGradients.Length; l++)
                for (int n = 0; n < weightGradients[l].Length; n++)
                    for (int w = 0; w < weightGradients[l][n].Length; w++)
                        weightGradients[l][n][w] += RegularizationFactor * Network[l][n][w];
        }
    }
}