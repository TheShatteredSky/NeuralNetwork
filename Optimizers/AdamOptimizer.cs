namespace NeuralNetwork.Optimizers;

public sealed class AdamOptimizer : SGDOptimizer
{
    private const double _decayRateOfFirstMoment = 0.9;
    private const double _decayRateOfSecondMoment = 0.999;
    private const double _epsilon = 0.00000001;
    private double[][][] _weightFirstMoments;
    private double[][] _biasFirstMoments;
    private double[][][] _weightSecondMoments;
    private double[][] _biasSecondMoments;
    private double _iteration;
    
    public AdamOptimizer(Network network, LossType lossFunction, double baseLearningRate) : base (network, lossFunction, baseLearningRate)
    {
        _weightFirstMoments = NetworkUtilities.InstantiateWeightArray(network);
        _biasFirstMoments = NetworkUtilities.InstantiateBiasArray(network);
        _weightSecondMoments = NetworkUtilities.InstantiateWeightArray(network);
        _biasSecondMoments = NetworkUtilities.InstantiateBiasArray(network);
        _iteration = 1;
    }

    public override void Optimize(double[][] inputs, double[][] outputs, uint totalEpochs)
    {
        var scaled = Network.ScaledData(inputs, outputs);
        inputs = scaled.Item1;
        outputs = scaled.Item2;
        double[][][] weightGradientsForBatch = NetworkUtilities.InstantiateWeightArray(Network);
        double[][] biasGradientsForBatch = NetworkUtilities.InstantiateBiasArray(Network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(inputs, outputs, weightGradientsForBatch, biasGradientsForBatch);
            if (epoch % 100 == 0 && epoch > 0) LearningRate *= 0.9995;
        }
    }

    public override double[] OptimizeTracked(double[][] inputs, double[][] outputs, uint totalEpochs)
    {
        var scaled = Network.ScaledData(inputs, outputs);
        inputs = scaled.Item1;
        outputs = scaled.Item2;
        List<double> tracker = new List<double>();
        tracker.Add(Network.Loss(inputs, outputs, LossType));
        double[][][] weightGradientsForBatch = NetworkUtilities.InstantiateWeightArray(Network);
        double[][] biasGradientsForBatch = NetworkUtilities.InstantiateBiasArray(Network);
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            ExecuteEpoch(inputs, outputs, weightGradientsForBatch, biasGradientsForBatch);
            if (epoch % 100 == 0 && epoch > 0)
            {
                tracker.Add(Network.Loss(inputs, outputs, LossType));
                LearningRate *= 0.9995;
            }
        }
        return tracker.ToArray();
    }

    private void ExecuteEpoch(double[][] inputs, double[][] outputs, double[][][] weightGradientsForBatch, double[][] biasGradientsForBatch)
    {
        int sampleCount = outputs.Length;
        int batchSize = 64;
        int batches = (sampleCount + batchSize - 1) / batchSize;
        for (int batchIndex = 0; batchIndex < batches; batchIndex++)
        {
            NetworkUtilities.ClearWeightArray(weightGradientsForBatch);
            NetworkUtilities.ClearBiasArray(biasGradientsForBatch);
            for (int sampleIndex = batchIndex * batchSize; sampleIndex < sampleCount && sampleIndex < batchIndex * batchSize + batchSize; sampleIndex++)
            {
                double[][][] weightGradientsPerLayer = NetworkUtilities.InstantiateWeightArray(Network);
                double[][] biasGradientsPerLayer = NetworkUtilities.InstantiateBiasArray(Network);
                NodeActivationRecord[][] forwardPassRecords = new NodeActivationRecord[Network.GetLayerCount()][];
                double[] layerInputs = new double[inputs[sampleIndex].Length];
                for (int i = 0; i < inputs[sampleIndex].Length; i++)
                    layerInputs[i] = inputs[sampleIndex][i];
                double[] predictions = ForwardPass(forwardPassRecords, layerInputs);
                double[] nextLayerDeltas = OutputLayerGradients(predictions, outputs, sampleIndex);
                BackPropagation(nextLayerDeltas, forwardPassRecords, weightGradientsPerLayer, biasGradientsPerLayer);
                AccumulateGradients(weightGradientsForBatch, biasGradientsForBatch, weightGradientsPerLayer, biasGradientsPerLayer, Network);
            }
            int currentBatchSize = Math.Min(64, sampleCount - batchSize * (batchIndex - 1));
            ScaleGradients(weightGradientsForBatch, biasGradientsForBatch, currentBatchSize);
            UpdateMoments(weightGradientsForBatch, biasGradientsForBatch);
            ApplyGradients(weightGradientsForBatch, biasGradientsForBatch, batchSize, Network, LearningRate);
        }
    }

    private void ScaleGradients(double[][][] weightGradientsForBatch, double[][] biasGradientsForBatch, int size)
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
    
    private void UpdateMoments(double[][][] weightGradients, double[][] biasGradients)
    {
        for (int layerIndex = 0; layerIndex < weightGradients.Length; layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < weightGradients[layerIndex].Length; nodeIndex++)
            {
                for (int weightIndex = 0; weightIndex < weightGradients[layerIndex][nodeIndex].Length; weightIndex++)
                {
                    double weightGradient = weightGradients[layerIndex][nodeIndex][weightIndex];
                    _weightFirstMoments[layerIndex][nodeIndex][weightIndex] = _decayRateOfFirstMoment * _weightFirstMoments[layerIndex][nodeIndex][weightIndex] + (1 - _decayRateOfFirstMoment) * weightGradient;
                    _weightSecondMoments[layerIndex][nodeIndex][weightIndex] = _decayRateOfSecondMoment * _weightSecondMoments[layerIndex][nodeIndex][weightIndex] + (1 - _decayRateOfSecondMoment) * weightGradient * weightGradient;
                }
                double biasGradient = biasGradients[layerIndex][nodeIndex];
                _biasFirstMoments[layerIndex][nodeIndex] = _decayRateOfFirstMoment * _biasFirstMoments[layerIndex][nodeIndex] + (1 - _decayRateOfFirstMoment) * biasGradient;
                _biasSecondMoments[layerIndex][nodeIndex] = _decayRateOfSecondMoment * _biasSecondMoments[layerIndex][nodeIndex] + (1 - _decayRateOfSecondMoment) * biasGradient * biasGradient;
            }
        }
    }

    protected override void ApplyGradients(double[][][] totalWeightGradients, double[][] totalBiasGradients, int sampleCount, Network network, double learningRate)
    {
        for (int layerIndex = 0; layerIndex < Network.GetLayerCount(); layerIndex++)
        {
            for (int nodeIndex = 0; nodeIndex < Network[layerIndex].GetSize(); nodeIndex++)
            {
                for (int weightIndex = 0; weightIndex < Network[layerIndex, nodeIndex].GetDimensions(); weightIndex++)
                {
                    double correctedWeightFirstMoment = _weightFirstMoments[layerIndex][nodeIndex][weightIndex] / (1 - Math.Pow(_decayRateOfFirstMoment, _iteration));
                    double correctedWeightSecondMoment = _weightSecondMoments[layerIndex][nodeIndex][weightIndex] / (1 - Math.Pow(_decayRateOfSecondMoment, _iteration));
                    Network[layerIndex, nodeIndex, weightIndex] -= LearningRate * correctedWeightFirstMoment / (Math.Sqrt(correctedWeightSecondMoment) + _epsilon);
                }
                double correctedBiasFirstMoment = _biasFirstMoments[layerIndex][nodeIndex] / (1 - Math.Pow(_decayRateOfFirstMoment, _iteration));
                double correctedBiasSecondMoment = _biasSecondMoments[layerIndex][nodeIndex] / (1 - Math.Pow(_decayRateOfSecondMoment, _iteration));
                Network[layerIndex, nodeIndex, Network[layerIndex, nodeIndex].GetDimensions()] -= LearningRate * correctedBiasFirstMoment / (Math.Sqrt(correctedBiasSecondMoment) + _epsilon);
            }
        }
        _iteration++;
    }
}