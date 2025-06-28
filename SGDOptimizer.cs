namespace NeuralNetwork;

public class SGDOptimizer : Optimizer
{
    public SGDOptimizer(Network network, LossFunction lossFunction, double baseLearningRate) : base(network, lossFunction, baseLearningRate)
    {
        
    }
    public override void Optimize(double[][] inputs, double[][] outputs, uint totalEpochs)
    {
        double previousLoss = 1;
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            int sampleCount = outputs.Length;
            int batchSize = 64;
            int batches = (sampleCount + batchSize - 1) / batchSize;
            for (int batchIndex = 0; batchIndex < batches; batchIndex++)
            {
                double[][][] weightGradientsForBatch = NetworkUtilities.InstantiateWeightArray(Network);
                double[][] biasGradientsForBatch = NetworkUtilities.InstantiateBiasArray(Network);
                for (int sampleIndex = batchIndex * batchSize; sampleIndex < sampleCount && sampleIndex < batchIndex * batchSize + batchSize; sampleIndex++)
                {
                    double[][][] weightGradientsPerLayer = NetworkUtilities.InstantiateWeightArray(Network);
                    double[][] biasGradientsPerLayer = NetworkUtilities.InstantiateBiasArray(Network);
                    NodeActivationRecord[][] forwardPassRecords = new NodeActivationRecord[Network.GetLayerCount()][];
                    double[] layerInputs = new double[inputs[sampleIndex].Length];
                    for (int i = 0; i < inputs[sampleIndex].Length; i++)
                        layerInputs[i] = inputs[sampleIndex][i];
                    double[] predictions = ForwardPass(ref forwardPassRecords, layerInputs);
                    double[] nextLayerDeltas = OutputLayerGradients(predictions, outputs, sampleIndex);
                    BackPropagation(nextLayerDeltas, forwardPassRecords, weightGradientsPerLayer, biasGradientsPerLayer);
                    AccumulateGradients(ref weightGradientsForBatch, ref biasGradientsForBatch, weightGradientsPerLayer, biasGradientsPerLayer);
                }
                ApplyGradients(weightGradientsForBatch, biasGradientsForBatch, batchSize);
            }
            if (epoch % 100 == 0)
            {
                BaseLearningRate *= 0.995;
                double cur = Network.Loss(inputs, outputs);
                if (previousLoss - cur < 1e-6) break;
                previousLoss = cur;
            }
        }
    }

    private void BackPropagation(double[] nextLayerDeltas, NodeActivationRecord[][] forwardPassRecords, double[][][] weightGradientsPerLayer, double[][] biasGradientsPerLayer)
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

                // Calculate delta for current node
                double delta;
                if (layerIndex == Network.GetLayerCount() - 1) // Output layer
                    delta = nextLayerDeltas[nodeIndex];
                else // Hidden layers
                {
                    delta = 0;
                    Layer downstreamLayer = Network[layerIndex + 1];
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

    private double[] OutputLayerGradients(double[] layerInputs, double[][] outputs, int sampleIndex)
    {
        double[] nextLayerDeltas = new double[layerInputs.Length];
        Layer outputLayer = Network[Network.GetLayerCount() - 1];

        for (int i = 0; i < outputLayer.GetSize(); i++)
        {
            double output = layerInputs[i];
            Node node = outputLayer.GetNodes()[i];

            switch (LossFunct)
            {
                case LossFunction.CrossEntropy:
                    if (node.GetActivation() == Node.ActivationType.Sigmoid || node.GetActivation() == Node.ActivationType.Softmax) nextLayerDeltas[i] = output - outputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
                case LossFunction.MSE:
                    if (node.GetActivation() == Node.ActivationType.Linear) nextLayerDeltas[i] = output - outputs[sampleIndex][i];
                    else throw new NotImplementedException();
                    break;
            }
        }
        return nextLayerDeltas;
    }

    private double[] ForwardPass(ref NodeActivationRecord[][] forwardPassRecords, double[] layerInputs)
    {
        for (int l = 0; l < Network.GetLayerCount(); l++)
        {
            Layer layer = Network[l];
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

    private NodeActivationRecord NodeRecord(Node node, double[] weightedSums, double[]? softmaxOutputs, int n, Layer layer, double[] inputs)
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
    

    private void AccumulateGradients(ref double[][][] weightAccumulator, ref double[][] biasAccumulator, double[][][] weights, double[][] biases)
    {
        for (int l = 0; l < Network.GetLayerCount(); l++)
        {
            for (int n = 0; n < Network[l].GetSize(); n++)
            {
                biasAccumulator[l][n] += biases[l][n];
                for (int w = 0; w < weights[l][n].Length; w++)
                {
                    weightAccumulator[l][n][w] += weights[l][n][w];
                }
            }
        }
    }
    
   void ApplyGradients(double[][][] totalWeightGradients, double[][] totalBiasGradients, int sampleCount)
    {
        for (int layerIndex = 0; layerIndex < Network.GetLayerCount(); layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < Network[layerIndex].GetSize(); nodeIndex++)
            {
                Network[layerIndex, nodeIndex, -1] -= BaseLearningRate * totalBiasGradients[layerIndex][nodeIndex] / sampleCount;
                for (int weightIndex = 0; weightIndex < Network[layerIndex, nodeIndex].GetDimensions(); weightIndex++)
                    Network[layerIndex, nodeIndex, weightIndex] -= BaseLearningRate * totalWeightGradients[layerIndex][nodeIndex][weightIndex] / sampleCount;
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