namespace NeuralNetwork.Optimizers;

/// <summary>
/// An instance of an Adaptive Moment Estimation Gradient Descent Optimizer.
/// </summary>
public sealed class AdamOptimizer : SGDOptimizer
{
    private const double _decayRateOfFirstMoment = 0.9;
    private const double _decayRateOfSecondMoment = 0.999;
    private const double _epsilon = 0.00000001;
    private double _mutableDecayRateOfFirstMoment;
    private double _mutableDecayRateOfSecondMoment;
    private double[][][] _weightFirstMoments;
    private double[][] _biasFirstMoments;
    private double[][][] _weightSecondMoments;
    private double[][] _biasSecondMoments;
    
    /// <summary>
    /// Creates an AdamOptimizer.
    /// </summary>
    /// <param name="network">The Network this AdamOptimizer is based on.</param>
    /// <param name="lossFunction">The loss function.</param>
    /// <param name="baseLearningRate">The starting learning rate.</param>
    public AdamOptimizer(Network network, LossType lossFunction, double baseLearningRate) : base (network, lossFunction, baseLearningRate)
    {
        _weightFirstMoments = Utilities.InstantiateWeightArray(network);
        _biasFirstMoments = Utilities.InstantiateBiasArray(network);
        _weightSecondMoments = Utilities.InstantiateWeightArray(network);
        _biasSecondMoments = Utilities.InstantiateBiasArray(network);
        _mutableDecayRateOfFirstMoment = _decayRateOfFirstMoment;
        _mutableDecayRateOfSecondMoment = _decayRateOfSecondMoment;
    }
    
    /// <summary>
    /// Optimizes the associated Network.
    /// </summary>
    /// <param name="data">The data to utilize.</param>
    /// <param name="totalEpochs">The number of epochs.</param>
    public override void Optimize(Dataset data, uint totalEpochs)
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
    public override double[] OptimizeTracked(Dataset data, uint totalEpochs)
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
                tracker.Add(Network.Loss(inputs, outputs, LossType));
                LearningRate *= 0.9995;
            }
        }
        return tracker.ToArray();
    }

    //Note: At no point should the data arrays be modified or returned, they should only be read.
    protected override void ExecuteEpoch(double[][][] weightGradientsForBatch, double[][] biasGradientsForBatch, double[][] inputs, double[][] outputs)
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
                double decayRateOfFirstMomentReciprocal = 1 - _mutableDecayRateOfFirstMoment;
                double decayRateOfSecondMomentReciprocal = 1 - _mutableDecayRateOfSecondMoment;
                for (int weightIndex = 0; weightIndex < Network[layerIndex, nodeIndex].GetSize(); weightIndex++)
                {
                    double correctedWeightFirstMoment = _weightFirstMoments[layerIndex][nodeIndex][weightIndex] / decayRateOfFirstMomentReciprocal;
                    double correctedWeightSecondMoment = _weightSecondMoments[layerIndex][nodeIndex][weightIndex] / decayRateOfSecondMomentReciprocal;
                    Network[layerIndex, nodeIndex, weightIndex] -= LearningRate * correctedWeightFirstMoment / (Math.Sqrt(correctedWeightSecondMoment) + _epsilon);
                }
                double correctedBiasFirstMoment = _biasFirstMoments[layerIndex][nodeIndex] / decayRateOfFirstMomentReciprocal;
                double correctedBiasSecondMoment = _biasSecondMoments[layerIndex][nodeIndex] / decayRateOfSecondMomentReciprocal;
                Network[layerIndex, nodeIndex, Network[layerIndex, nodeIndex].GetSize()] -= LearningRate * correctedBiasFirstMoment / (Math.Sqrt(correctedBiasSecondMoment) + _epsilon);
            }
        }
        _mutableDecayRateOfFirstMoment *= _decayRateOfFirstMoment;
        _mutableDecayRateOfSecondMoment *=  _decayRateOfSecondMoment;
    }
}