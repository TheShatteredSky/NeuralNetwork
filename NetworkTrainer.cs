namespace NeuralNetwork;

public static class NetworkTrainer
{
    public static void Optimize(Network network, double[][] features, double[][] expectedOutputs, uint totalEpochs)
    {
        double previousLoss = 1;
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            int sampleCount = expectedOutputs.Length;
            int batchSize = 64;
            int batches = (sampleCount + batchSize - 1) / batchSize;
            for (int batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                double[][][] weightGradientsForBatch = NetworkUtilities.InstantiateWeightArray(network);
                double[][] biasGradientsForBatch = NetworkUtilities.InstantiateBiasArray(network);
                for (int sampleIndex = batchIndex * batchSize; sampleIndex < sampleCount && sampleIndex < batchIndex * batchSize + batchSize; sampleIndex++)
                {
                    double[][][] weightGradientsPerLayer = NetworkUtilities.InstantiateWeightArray(network);
                    double[][] biasGradientsPerLayer = NetworkUtilities.InstantiateBiasArray(network);
                    NodeActivationRecord[][] forwardPassRecords = new NodeActivationRecord[network.GetLayerCount()][];
                    double[] layerInputs = new double[features[sampleIndex].Length];
                    for (int i = 0; i < features[sampleIndex].Length; i++)
                        layerInputs[i] = features[sampleIndex][i];
                    double[] predictions = ForwardPass(network, ref forwardPassRecords, layerInputs);
                    double[] nextLayerDeltas = OutputLayerGradients(network, predictions, expectedOutputs, sampleIndex);
                    BackPropagation(network, nextLayerDeltas, forwardPassRecords, weightGradientsPerLayer, biasGradientsPerLayer);
                    AccumulateGradients(network,ref weightGradientsForBatch, ref biasGradientsForBatch, weightGradientsPerLayer, biasGradientsPerLayer);
                }
                ApplyGradients(network, weightGradientsForBatch, biasGradientsForBatch, batchSize);
            }
            if (epoch % 100 == 0)
            {
                network.SetLearningRate(network.GetLearningRate() * 0.995);
                double cur = network.Loss(features, expectedOutputs);
                if (previousLoss - cur < 1e-6) break;
                previousLoss = cur;
            }
        }
    }

    private static void BackPropagation(Network network, double[] nextLayerDeltas, NodeActivationRecord[][] forwardPassRecords, double[][][] weightGradientsPerLayer, double[][] biasGradientsPerLayer)
    {

        for (int layerIndex = network.GetLayerCount() - 1; layerIndex >= 0; layerIndex--)
        {
            Layer currentLayer = network[layerIndex];
            NodeActivationRecord[] layerActivationRecords = forwardPassRecords[layerIndex];
            double[] currentLayerDeltas = new double[currentLayer.GetSize()];

            for (int nodeIndex = 0; nodeIndex < currentLayer.GetSize(); nodeIndex++)
            {
                Node currentNode = currentLayer.GetNodes()[nodeIndex];
                NodeActivationRecord activationRecord = layerActivationRecords[nodeIndex];

                // Calculate delta for current node
                double delta;
                if (layerIndex == network.GetLayerCount() - 1) // Output layer
                    delta = nextLayerDeltas[nodeIndex];
                else // Hidden layers
                {
                    delta = 0;
                    Layer downstreamLayer = network[layerIndex + 1];
                    foreach (Node downstreamNode in downstreamLayer.GetNodes())
                        for (int parentIndex = 0; parentIndex < downstreamNode.GetParents().Length; parentIndex++)
                            if (downstreamNode.GetParents()[parentIndex] == nodeIndex)
                                delta += downstreamNode.GetWeights()[parentIndex] * nextLayerDeltas[Array.IndexOf(downstreamLayer.GetNodes(), downstreamNode)];
                    delta *= activationRecord.ActivationDerivative;
                }
                currentLayerDeltas[nodeIndex] = delta;

                for (int weightIndex = 0; weightIndex < currentNode.GetDimensions(); weightIndex++)
                    weightGradientsPerLayer[layerIndex][nodeIndex][weightIndex] +=
                        delta * activationRecord.InputValues[weightIndex];
                biasGradientsPerLayer[layerIndex][nodeIndex] += delta;
            }
            nextLayerDeltas = currentLayerDeltas;
        }
    }

    private static double[] OutputLayerGradients(Network network, double[] layerInputs, double[][] expectedOutputs, int sampleIndex)
    {
        double[] nextLayerDeltas = new double[layerInputs.Length];
        Layer outputLayer = network[network.GetLayerCount() - 1];

        for (int i = 0; i < outputLayer.GetSize(); i++)
        {
            double output = layerInputs[i];
            Node node = outputLayer.GetNodes()[i];

            switch (network.GetLossFunction())
            {
                case Network.LossFunction.CrossEntropy:
                    if (node.GetActivation() == Node.ActivationType.Sigmoid || node.GetActivation() == Node.ActivationType.Softmax) nextLayerDeltas[i] = output - expectedOutputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
                case Network.LossFunction.MSE:
                    if (node.GetActivation() == Node.ActivationType.Linear) nextLayerDeltas[i] = output - expectedOutputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
            }
        }
        return nextLayerDeltas;
    }

    private static double[] ForwardPass(Network network, ref NodeActivationRecord[][] forwardPassRecords, double[] layerInputs)
    {
        for (int l = 0; l < network.GetLayerCount(); l++)
        {
            Layer layer = network[l];
            Node[] nodes = layer.GetNodes();
            NodeActivationRecord[] layerRecord = new NodeActivationRecord[layer.GetSize()];
            double[] layerOutputs = new double[layer.GetSize()];
            double[] weightedSums = layer.WeightedSums(layerInputs);
            double[]? softmaxOutputs = layer[0].GetActivation() == Node.ActivationType.Softmax ? layer.SoftmaxOutputs(weightedSums) : null;
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

    private static NodeActivationRecord NodeRecord(Node node, double[] weightedSums, double[]? softmaxOutputs, int n, Layer layer, double[] inputs)
    {
        double activationOutput = 0;
        double activationDerivative = 0;
        switch (node.GetActivation())
        {
            case Node.ActivationType.Softmax:
                activationOutput = softmaxOutputs![n];
                activationDerivative = 1;
                break;
            case Node.ActivationType.Sigmoid:
                activationOutput = Functions.Sigmoid(weightedSums[n]);
                activationDerivative = activationOutput * (1 - activationOutput);
                break;
            case Node.ActivationType.Linear:
                activationOutput = weightedSums[n];
                activationDerivative = 1;
                break;
            case Node.ActivationType.RElu:
                activationOutput = Math.Max(0, weightedSums[n]);
                activationDerivative = weightedSums[n] > 0 ? 1 : 0;
                break;
            case Node.ActivationType.LeakyRElu:
                activationOutput = Math.Max(0, weightedSums[n]);
                activationDerivative = weightedSums[n] > 0 ? 1 : 0.01;
                break;
            case Node.ActivationType.Tanh:
                activationOutput = Math.Tanh(weightedSums[n]);
                activationDerivative = 1 - activationOutput * activationOutput;
                break;
            // TODO: Other derivatives
        }
        return new NodeActivationRecord { InputValues = layer.NodeInputs(inputs, n), WeightedSum = weightedSums[n], ActivationOutput = activationOutput, ActivationDerivative = activationDerivative };
    }
    

    private static void AccumulateGradients(Network network, ref double[][][] weightAccumulator, ref double[][] biasAccumulator, double[][][] weights, double[][] biases)
    {
        for (int l = 0; l < network.GetLayerCount(); l++)
        {
            for (int n = 0; n < network[l].GetSize(); n++)
            {
                biasAccumulator[l][n] += biases[l][n];
                for (int w = 0; w < weights[l][n].Length; w++)
                {
                    weightAccumulator[l][n][w] += weights[l][n][w];
                }
            }
        }
    }

    private static void ApplyGradients(Network network, double[][][] totalWeightGradients, double[][] totalBiasGradients, int sampleCount)
    {
        for (int layerIndex = 0; layerIndex < network.GetLayerCount(); layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < network[layerIndex].GetSize(); nodeIndex++)
            {
                network[layerIndex, nodeIndex, -1] -= network.GetLearningRate() * totalBiasGradients[layerIndex][nodeIndex] / sampleCount;
                for (int weightIndex = 0; weightIndex < network[layerIndex, nodeIndex].GetDimensions(); weightIndex++)
                    network[layerIndex, nodeIndex, weightIndex] -= network.GetLearningRate() * totalWeightGradients[layerIndex][nodeIndex][weightIndex] / sampleCount;
            }
        }
    }

    private class NodeActivationRecord
    {
        public required double[] InputValues { get; init; } // Inputs received from previous layer
        public double WeightedSum { get; set; } // z = sum(weights * inputs) + bias
        public double ActivationOutput { get; set; } // a = activation(z)
        public double ActivationDerivative { get; init; } // da/dz derivative at z
    }
}
//