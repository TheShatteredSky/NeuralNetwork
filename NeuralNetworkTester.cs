using System.Globalization;
using NeuralNetwork;
using System.Text;
using System.Diagnostics;

/*public class NeuralNetworkTester
{
    private const string TestFileName = "network_test.nnet";
    private const string TestDataFile = "test_data.csv";
    private const string PerformanceFile = "performance_log.txt";
    
    public static void RunAllTests()
    {
        TestNetworkConstruction();
        TestWeightInitialization();
        TestInputIntegrity();
        TestForwardPass();
        TestActivationFunctions();
        TestTraining();
        TestSerialization();
        TestNetworkUtilities();
        TestNetworkManager();
        TestEdgeCases();
        TestPerformance();
        TestConcurrency();
        TestLoadCompleteness();
        TestVectorizedOperations();
        TestBackpropagation();
        TestDataNormalization();
        
        Console.WriteLine("All tests passed successfully!");
    }

    public static void TestNetworkConstruction()
    {
        // Test basic network creation
        Network network = new Network("ConstructionTest");
        network.InstantiateBasics(1, Network.LossFunction.MSE, 0.01);
        network.CreateInputLayer(2, Node.ActivationType.Linear);
        network.CreateHiddenLayers(3, Node.ActivationType.RElu);
        network.CreateOutputLayer(1, Node.ActivationType.Sigmoid);
        
        // Validate layer counts
        if (network.GetLayerCount() != 3)
            throw new Exception("Layer count mismatch");
        
        // Validate node counts
        if (network[0].GetSize() != 2 || 
            network[1].GetSize() != 3 || 
            network[2].GetSize() != 1)
            throw new Exception("Node count mismatch");
        
        // Validate activation types
        if (network[0, 0].GetActivation() != Node.ActivationType.Linear ||
            network[1, 0].GetActivation() != Node.ActivationType.RElu ||
            network[2, 0].GetActivation() != Node.ActivationType.Sigmoid)
            throw new Exception("Activation type mismatch");
        
        Console.WriteLine("Network construction test passed");
    }

    public static void TestWeightInitialization()
    {
        Network network = new Network("WeightInitTest");
        network.InstantiateBasics(1, Network.LossFunction.MSE, 0.01);
        network.CreateInputLayer(2, Node.ActivationType.Linear);
        network.CreateHiddenLayers(3, Node.ActivationType.RElu);
        network.CreateOutputLayer(1, Node.ActivationType.Sigmoid);
        
        // Test default initialization
        network.Randomize(0);
        
        // Validate He initialization for ReLU
        double sum = 0;
        for (int i = 0; i < network[1].GetSize(); i++)
        {
            for (int j = 0; j < network[1, i].GetDimensions(); j++)
            {
                double weight = network[1, i, j];
                sum += weight * weight;
            }
        }
        double variance = sum / (3 * 2); // 3 neurons, 2 weights each
        double expectedStd = Math.Sqrt(2.0 / 2);
        
        if (Math.Abs(Math.Sqrt(variance) - expectedStd) > 0.5)
            throw new Exception("He initialization variance mismatch");
        
        Console.WriteLine("Weight initialization test passed");
    }

    public static void TestInputIntegrity()
    {
        Network network = new Network("InputIntegrityTest");
        network.InstantiateBasics(1, Network.LossFunction.MSE, 0.01);
        network.CreateInputLayer(2, Node.ActivationType.Linear);
        network.CreateHiddenLayers(3, Node.ActivationType.RElu);
        network.CreateOutputLayer(1, Node.ActivationType.Sigmoid);
        network.Randomize(0.5);
        
        // Create test input
        double[][] originalInputs = { new double[] { 0.5, 0.7 } };
        double[][] inputsCopy = { new double[] { 0.5, 0.7 } };
        
        // Process data
        network.Process(originalInputs);
        
        // Verify original inputs unchanged
        for (int i = 0; i < inputsCopy.Length; i++)
        {
            for (int j = 0; j < inputsCopy[i].Length; j++)
            {
                if (Math.Abs(originalInputs[i][j] - inputsCopy[i][j]) > 1e-9)
                    throw new Exception("Input data was modified during processing");
            }
        }
        
        Console.WriteLine("Input integrity test passed");
    }

    public static void TestForwardPass()
    {
        Network network = new Network("ForwardPassTest");
        network.InstantiateBasics(1, Network.LossFunction.MSE, 0.01);
        network.CreateInputLayer(2, Node.ActivationType.Linear);
        network.CreateHiddenLayers(2, Node.ActivationType.Linear);
        network.CreateOutputLayer(1, Node.ActivationType.Linear);
        
        // Set known weights
        network[0, 0].SetWeights(new double[] { 1.0 });
        network[0, 0].SetBias(0);
        network[0, 1].SetWeights(new double[] { 1.0 });
        network[0, 1].SetBias(0);
        
        network[1, 0].SetWeights(new double[] { 0.5, 0.5 });
        network[1, 0].SetBias(0);
        network[1, 1].SetWeights(new double[] { 0.5, 0.5 });
        network[1, 1].SetBias(0);
        
        network[2, 0].SetWeights(new double[] { 1.0, 1.0 });
        network[2, 0].SetBias(0);
        
        // Test calculation
        double[][] output = network.Process(new double[][] { new double[] { 1.0, 2.0 } });
        
        // Expected: (1*0.5 + 2*0.5) + (1*0.5 + 2*0.5) = 1.5 + 1.5 = 3.0
        if (Math.Abs(output[0][0] - 3.0) > 1e-9)
            throw new Exception("Forward pass calculation error");
        
        Console.WriteLine("Forward pass test passed");
    }

    public static void TestTraining()
    {
        // XOR dataset
        double[][] inputs = {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        
        double[][] outputs = {
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 }
        };
        
        // Create network
        Network network = new Network("TrainingTest");
        network.InstantiateBasics(1, Network.LossFunction.MSE, 0.1);
        network.CreateInputLayer(2, Node.ActivationType.Linear);
        network.CreateHiddenLayers(2, Node.ActivationType.Sigmoid);
        network.CreateOutputLayer(1, Node.ActivationType.Sigmoid);
        network.Randomize(0.5);
        
        // Measure initial loss
        double initialLoss = network.Loss(inputs, outputs);
        
        // Train network
        network.Optimize(inputs, outputs, 1000, true);
        
        // Measure final loss
        double finalLoss = network.Loss(inputs, outputs);
        
        // Verify significant improvement
        if (finalLoss >= initialLoss || finalLoss > 0.1)
            throw new Exception($"Training failed: Initial loss={initialLoss}, Final loss={finalLoss}");
        
        // Verify predictions
        double[][] predictions = network.Process(inputs);
        for (int i = 0; i < outputs.Length; i++)
        {
            double expected = outputs[i][0];
            double predicted = predictions[i][0];
            if (Math.Abs(predicted - expected) > 0.3)
                throw new Exception($"Prediction error: Expected {expected}, got {predicted}");
        }
        
        Console.WriteLine("Training test passed");
    }

    public static void TestSerialization()
    {
        // Create original network
        Network original = new Network("SerializationTest");
        original.InstantiateBasics(1, Network.LossFunction.CrossEntropy, 0.05);
        original.CreateInputLayer(3, Node.ActivationType.RElu);
        original.CreateHiddenLayers(4, Node.ActivationType.Sigmoid);
        original.CreateOutputLayer(2, Node.ActivationType.Softmax);
        original.Randomize(0.5);
        
        // Save to string
        string serialized = original.ToString();
        
        // Save to file
        NetworkUtilities.SaveToFile(TestFileName, original);
        
        // Load from file
        Network loaded = NetworkUtilities.LoadFromFile(TestFileName);
        
        // Compare properties
        if (original.GetName() != loaded.GetName())
            throw new Exception("Name serialization mismatch");
        
        if (original.GetLayerCount() != loaded.GetLayerCount())
            throw new Exception("Layer count serialization mismatch");
        
        // Compare weights and biases
        for (int l = 0; l < original.GetLayerCount(); l++)
        {
            for (int n = 0; n < original[l].GetSize(); n++)
            {
                // Compare biases
                if (Math.Abs(original[l, n, -1] - loaded[l, n, -1]) > 1e-9)
                    throw new Exception("Bias serialization mismatch");
                
                // Compare weights
                for (int w = 0; w < original[l, n].GetDimensions(); w++)
                {
                    if (Math.Abs(original[l, n, w] - loaded[l, n, w]) > 1e-9)
                        throw new Exception("Weight serialization mismatch");
                }
                
                // Compare activation
                if (original[l, n].GetActivation() != loaded[l, n].GetActivation())
                    throw new Exception("Activation serialization mismatch");
            }
        }
        
        Console.WriteLine("Serialization test passed");
    }

    public static void TestNetworkUtilities()
    {
        // Test data normalization
        double[][] data = {
            new double[] { 10, 1000 },
            new double[] { 20, 2000 },
            new double[] { 30, 3000 }
        };
        
        NetworkUtilities.NormalizeData(data);
        
        if (data[0][0] != 0 || data[0][1] != 0 ||
            data[1][0] != 0.5 || data[1][1] != 0.5 ||
            data[2][0] != 1 || data[2][1] != 1)
            throw new Exception("Data normalization failed");
        
        // Test random number generation
        double randomValue = NetworkUtilities.RandomDouble(5, 10);
        if (randomValue < 5 || randomValue > 10)
            throw new Exception("Random number generation failed");
        
        Console.WriteLine("Network utilities test passed");
    }

    public static void TestNetworkManager()
    {
        // Create base network
        Network baseNetwork = new Network("ManagerBase");
        baseNetwork.InstantiateBasics(1, Network.LossFunction.MSE, 0.1);
        baseNetwork.CreateInputLayer(2, Node.ActivationType.Linear);
        baseNetwork.CreateHiddenLayers(3, Node.ActivationType.RElu);
        baseNetwork.CreateOutputLayer(1, Node.ActivationType.Sigmoid);
        
        // Create manager
        NetworkManager manager = new NetworkManager(baseNetwork);
        
        // XOR dataset
        double[][] inputs = {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        
        double[][] outputs = {
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 }
        };
        
        // Test network generation
        Network generated = manager.GenerateNetwork();
        if (generated.GetLayerCount() != baseNetwork.GetLayerCount())
            throw new Exception("Network generation layer count mismatch");
        
        // Test FindBest
        Network best = manager.FindBest(inputs, outputs, 1, 500, 4);
        double loss = best.Loss(inputs, outputs);
        if (loss > 0.25)
            throw new Exception($"FindBest produced poor network (loss={loss})");
        
        // Test GenerateScores
        double[] scores = manager.GenerateScores(inputs, outputs, 1, 500, 4);
        if (scores.Length == 0 || scores[0] > 0.3)
            throw new Exception("GenerateScores produced unexpected results");
        
        Console.WriteLine("Network manager test passed");
    }

    public static void TestActivationFunctions()
    {
        Console.WriteLine("\n--- Activation Function Tests ---");
        
        // Test all activation functions with known values
        var testCases = new (Node.ActivationType, double, double, double)[]
        {
            (Node.ActivationType.Sigmoid, 0, 0.5, 0.25),
            (Node.ActivationType.Tanh, 0, 0, 1),
            (Node.ActivationType.RElu, -1, 0, 0),
            (Node.ActivationType.RElu, 1, 1, 1),
            (Node.ActivationType.LeakyRElu, -1, -0.01, 0.01),
            (Node.ActivationType.LeakyRElu, 1, 1, 1),
            (Node.ActivationType.Linear, 2.5, 2.5, 1),
            (Node.ActivationType.Softmax, 1, double.NaN, double.NaN) // Special case
        };

        foreach (var (activation, input, expectedOutput, expectedDerivative) in testCases)
        {
            Node node = new Node(0, 0, 1, activation, null);
            
            // Test forward pass
            double output = node.Process([input]);
            
            // Test derivative calculation
            double derivative = 0;
            switch (activation)
            {
                case Node.ActivationType.Sigmoid:
                    derivative = output * (1 - output);
                    break;
                case Node.ActivationType.Tanh:
                    derivative = 1 - output * output;
                    break;
                case Node.ActivationType.RElu:
                    derivative = input > 0 ? 1 : 0;
                    break;
                case Node.ActivationType.LeakyRElu:
                    derivative = input > 0 ? 1 : 0.01;
                    break;
                case Node.ActivationType.Linear:
                    derivative = 1;
                    break;
            }
            
            if (activation != Node.ActivationType.Softmax)
            {
                if (Math.Abs(output - expectedOutput) > 1e-9)
                    throw new Exception($"{activation} output mismatch: Expected {expectedOutput}, got {output}");
                
                if (Math.Abs(derivative - expectedDerivative) > 1e-9)
                    throw new Exception($"{activation} derivative mismatch: Expected {expectedDerivative}, got {derivative}");
            }
            else
            {
                // Special handling for softmax
                if (!double.IsNaN(output))
                    throw new Exception("Softmax should only be processed at layer level");
            }
        }
        
        Console.WriteLine("Activation functions test passed");
    }

    public static void TestLoadCompleteness()
    {
        Console.WriteLine("\n--- Network Load Completeness Test ---");
        
        // Create original network
        Network original = new Network("LoadCompletenessTest");
        original.InstantiateBasics(2, Network.LossFunction.CrossEntropy, 0.05);
        original.CreateInputLayer(3, Node.ActivationType.RElu);
        original.CreateHiddenLayers(4, Node.ActivationType.Sigmoid);
        original.CreateOutputLayer(2, Node.ActivationType.Softmax);
        original.Randomize(0.5);
        
        // Save to file
        NetworkUtilities.SaveToFile(TestFileName, original);
        
        // Load from file
        Network loaded = NetworkUtilities.LoadFromFile(TestFileName);
        
        // Verify complete structure
        if (original.GetName() != loaded.GetName())
            throw new Exception("Name mismatch after loading");
        
        if (original.GetLayerCount() != loaded.GetLayerCount())
            throw new Exception("Layer count mismatch after loading");
        
        if (original.GetLossFunction() != loaded.GetLossFunction())
            throw new Exception("Loss function mismatch after loading");
        
        if (Math.Abs(original.GetLearningRate() - loaded.GetLearningRate()) > 1e-9)
            throw new Exception("Learning rate mismatch after loading");
        
        // Verify layer types and sizes
        for (int l = 0; l < original.GetLayerCount(); l++)
        {
            var origLayer = original[l];
            var loadedLayer = loaded[l];
            
            if (origLayer.GetLayerIdentifier() != loadedLayer.GetLayerIdentifier())
                throw new Exception($"Layer ID mismatch at layer {l}");
            
            if (origLayer.GetSize() != loadedLayer.GetSize())
                throw new Exception($"Node count mismatch at layer {l}");
        }
        
        // Verify all nodes
        for (int l = 0; l < original.GetLayerCount(); l++)
        {
            for (int n = 0; n < original[l].GetSize(); n++)
            {
                var origNode = original[l, n];
                var loadedNode = loaded[l, n];
                
                // Compare node properties
                if (origNode.GetIdentifier() != loadedNode.GetIdentifier())
                    throw new Exception($"Node ID mismatch at layer {l}, node {n}");
                
                if (origNode.GetDimensions() != loadedNode.GetDimensions())
                    throw new Exception($"Weight dimension mismatch at layer {l}, node {n}");
                
                if (origNode.GetActivation() != loadedNode.GetActivation())
                    throw new Exception($"Activation mismatch at layer {l}, node {n}");
                
                if (Math.Abs(origNode.GetBias() - loadedNode.GetBias()) > 1e-9)
                    throw new Exception($"Bias mismatch at layer {l}, node {n}");
                
                // Compare weights
                for (int w = 0; w < origNode.GetDimensions(); w++)
                {
                    if (Math.Abs(origNode.GetWeights()[w] - loadedNode.GetWeights()[w]) > 1e-9)
                        throw new Exception($"Weight mismatch at layer {l}, node {n}, weight {w}");
                }
                
                // Compare parents
                if (origNode.GetParents() != null && loadedNode.GetParents() != null)
                {
                    if (origNode.GetParents().Length != loadedNode.GetParents().Length)
                        throw new Exception($"Parent count mismatch at layer {l}, node {n}");
                    
                    for (int p = 0; p < origNode.GetParents().Length; p++)
                    {
                        if (origNode.GetParents()[p] != loadedNode.GetParents()[p])
                            throw new Exception($"Parent mismatch at layer {l}, node {n}, parent {p}");
                    }
                }
            }
        }
        
        // Verify functional equivalence
        double[][] testInput = { new double[] { 0.2, 0.4, 0.6 } };
        var originalOutput = original.Process(testInput);
        var loadedOutput = loaded.Process(testInput);
        
        for (int i = 0; i < originalOutput[0].Length; i++)
        {
            if (Math.Abs(originalOutput[0][i] - loadedOutput[0][i]) > 1e-9)
                throw new Exception($"Output mismatch at index {i}");
        }
        
        Console.WriteLine("Network load completeness test passed");
    }

    public static void TestVectorizedOperations()
    {
        Console.WriteLine("\n--- Vectorized Operations Test ---");
        
        // Create node with known weights
        Node node = new Node(0, 0, 4, Node.ActivationType.Linear, null);
        node.SetWeights(new double[] { 1.0, 2.0, 3.0, 4.0 });
        node.SetBias(0.5);
        
        // Test inputs
        double[] input = { 0.5, 1.0, 1.5, 2.0 };
        
        // Calculate expected result
        double expected = 0.5*1.0 + 1.0*2.0 + 1.5*3.0 + 2.0*4.0 + 0.5;
        
        // Get actual result
        double actual = node.Process(input);
        
        if (Math.Abs(expected - actual) > 1e-9)
            throw new Exception($"Vectorized dot product mismatch: Expected {expected}, got {actual}");
        
        // Test with non-vector-aligned input
        double[] input2 = { 0.5, 1.0, 1.5, 2.0, 2.5 };
        node = new Node(0, 0, 5, Node.ActivationType.Linear, null);
        node.SetWeights(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        expected = 0.5*1 + 1*2 + 1.5*3 + 2*4 + 2.5*5;
        actual = node.Process(input2);
        
        if (Math.Abs(expected - actual) > 1e-9)
            throw new Exception("Non-vector-aligned dot product mismatch");
        
        Console.WriteLine("Vectorized operations test passed");
    }

    public static void TestEdgeCases()
    {
        Console.WriteLine("\n--- Edge Case Tests ---");
        
        // Test empty network
        try
        {
            Network empty = new Network("EmptyTest");
            empty.Process(new double[][] { new double[] { 1.0 } });
            throw new Exception("Should throw exception for uninitialized network");
        }
        catch (NullReferenceException) { /*Expected*//* }
        
        Network single = new Network("SingleNode");
        single.InstantiateBasics(0, Network.LossFunction.MSE, 0.1);
        single.CreateInputLayer(1, Node.ActivationType.Linear);
        single.CreateOutputLayer(1, Node.ActivationType.Linear);
        single[0, 0].SetWeights(new double[] { 1.0 });
        single[0, 0].SetBias(0);
        single[1, 0].SetWeights(new double[] { 1.0 });
        single[1, 0].SetBias(0);
        
        double[][] output = single.Process(new double[][] { new double[] { 2.0 } });
        if (Math.Abs(output[0][0] - 2.0) > 1e-9)
            throw new Exception("Single-node network failed");
        
        // Test large value handling
        Network large = new Network("LargeValue");
        large.InstantiateBasics(1, Network.LossFunction.MSE, 0.1);
        large.CreateInputLayer(1, Node.ActivationType.RElu);
        large.CreateHiddenLayers(1, Node.ActivationType.RElu);
        large.CreateOutputLayer(1, Node.ActivationType.Linear);
        
        double[][] largeInput = { new double[] { 1000000.0 } };
        var largeOutput = large.Process(largeInput);
        if (double.IsNaN(largeOutput[0][0]))
            throw new Exception("Large input caused NaN output");
        Console.WriteLine("Edge cases test passed");
    }

    public static void TestBackpropagation()
    {
        Console.WriteLine("\n--- Backpropagation Test ---");
        
        // Create minimal network: 1 input -> 1 output
        Network network = new Network("BackpropTest");
        network.InstantiateBasics(0, Network.LossFunction.MSE, 0.5);
        network.CreateInputLayer(1, Node.ActivationType.Linear);
        network.CreateOutputLayer(1, Node.ActivationType.Linear);
        
        // Set known weights and biases
        network[0, 0].SetWeights(new double[] { 0.5 });
        network[0, 0].SetBias(0.2);
        network[1, 0].SetWeights(new double[] { 1.5 });
        network[1, 0].SetBias(0.3);
        
        // Training data
        double[][] inputs = { new double[] { 2.0 } };
        double[][] outputs = { new double[] { 3.0 } };
        
        // Store initial parameters
        double initialWeight1 = network[0, 0, 0];
        double initialBias1 = network[0, 0, -1];
        double initialWeight2 = network[1, 0, 0];
        double initialBias2 = network[1, 0, -1];
        
        // Run one optimization step
        network.Optimize(inputs, outputs, 1, false);
        
        // Verify weight updates (numerically)
        double predicted = network.Process(inputs)[0][0];
        double error = 3.0 - predicted;
        
        // Expected gradients (chain rule):
        // dE/dW2 = (dE/dy) * (dy/dW2) = 2*(y_pred - y_true) * input_hidden
        // dE/dW1 = (dE/dy) * (dy/dW2) * (dHidden/dW1) = 2*(y_pred - y_true) * W2 * input
        
        // Calculate expected changes
        double expectedDeltaW2 = -0.5 * 2 * error * (initialWeight1 * 2.0 + initialBias1);
        double expectedDeltaW1 = -0.5 * 2 * error * initialWeight2 * 2.0;
        double expectedDeltaB2 = -0.5 * 2 * error;
        double expectedDeltaB1 = -0.5 * 2 * error * initialWeight2;
        
        // Compare with actual changes
        double deltaW1 = network[0, 0, 0] - initialWeight1;
        double deltaB1 = network[0, 0, -1] - initialBias1;
        double deltaW2 = network[1, 0, 0] - initialWeight2;
        double deltaB2 = network[1, 0, -1] - initialBias2;
        
        if (Math.Abs(deltaW1 - expectedDeltaW1) > 1e-5)
            throw new Exception($"Weight1 update mismatch: Expected {expectedDeltaW1}, got {deltaW1}");
        
        if (Math.Abs(deltaB1 - expectedDeltaB1) > 1e-5)
            throw new Exception($"Bias1 update mismatch: Expected {expectedDeltaB1}, got {deltaB1}");
        
        if (Math.Abs(deltaW2 - expectedDeltaW2) > 1e-5)
            throw new Exception($"Weight2 update mismatch: Expected {expectedDeltaW2}, got {deltaW2}");
        
        if (Math.Abs(deltaB2 - expectedDeltaB2) > 1e-5)
            throw new Exception($"Bias2 update mismatch: Expected {expectedDeltaB2}, got {deltaB2}");
        
        Console.WriteLine("Backpropagation test passed");
    }

    public static void TestDataNormalization()
    {
        Console.WriteLine("\n--- Data Normalization Test ---");
        
        // Create test data
        double[][] data = {
            new double[] { 0, 100 },
            new double[] { 50, 200 },
            new double[] { 100, 300 }
        };
        
        // Create copy for comparison
        double[][] original = data.Select(a => a.ToArray()).ToArray();
        
        // Normalize
        NetworkUtilities.NormalizeData(data);
        
        // Verify original unchanged
        for (int i = 0; i < original.Length; i++)
        {
            for (int j = 0; j < original[i].Length; j++)
            {
                if (original[i][j] != data[i][j] + (j == 0 ? 0 : 100))
                    throw new Exception("Normalization modified original array");
            }
        }
        
        // Verify normalization
        if (data[0][0] != 0 || data[0][1] != 0 ||
            data[1][0] != 0.5 || data[1][1] != 0.5 ||
            data[2][0] != 1 || data[2][1] != 1)
            throw new Exception("Normalization produced incorrect values");
        
        // Test with constant column
        double[][] constantData = {
            new double[] { 5, 10 },
            new double[] { 5, 20 },
            new double[] { 5, 30 }
        };
        
        NetworkUtilities.NormalizeData(constantData);
        if (constantData[0][0] != 0 || constantData[1][0] != 0 || constantData[2][0] != 0)
            throw new Exception("Constant column not zeroed");
        
        Console.WriteLine("Data normalization test passed");
    }

    public static void TestPerformance()
    {
        Console.WriteLine("\n--- Performance Test ---");
        Stopwatch sw = new Stopwatch();
        
        // Create large network (100-50-10)
        Network largeNet = new Network("PerformanceTest");
        largeNet.InstantiateBasics(1, Network.LossFunction.MSE, 0.01);
        largeNet.CreateInputLayer(100, Node.ActivationType.RElu);
        largeNet.CreateHiddenLayers(50, Node.ActivationType.RElu);
        largeNet.CreateOutputLayer(10, Node.ActivationType.Sigmoid);
        largeNet.Randomize(0.5);
        
        // Generate large dataset
        double[][] bigInput = new double[1000][];
        double[][] bigOutput = new double[1000][];
        Random rand = new Random();
        
        for (int i = 0; i < 1000; i++)
        {
            bigInput[i] = new double[100];
            bigOutput[i] = new double[10];
            for (int j = 0; j < 100; j++) bigInput[i][j] = rand.NextDouble();
            for (int j = 0; j < 10; j++) bigOutput[i][j] = rand.NextDouble();
        }
        
        // Measure processing speed
        sw.Start();
        var output = largeNet.Process(bigInput);
        sw.Stop();
        Console.WriteLine($"Forward pass: {sw.ElapsedMilliseconds}ms for 1000 samples");
        
        // Measure training speed
        sw.Restart();
        largeNet.Optimize(bigInput, bigOutput, 1, false);
        sw.Stop();
        Console.WriteLine($"Training epoch: {sw.ElapsedMilliseconds}ms");
        
        // Append to performance log
        File.AppendAllText(PerformanceFile, 
            $"{DateTime.Now}: Forward={sw.ElapsedMilliseconds}ms, Training={sw.ElapsedMilliseconds}ms\n");
        
        Console.WriteLine("Performance test completed");
    }

    public static void TestConcurrency()
    {
        Console.WriteLine("\n--- Concurrency Test ---");
        
        Network network = new Network("ConcurrencyTest");
        network.InstantiateBasics(1, Network.LossFunction.MSE, 0.1);
        network.CreateInputLayer(2, Node.ActivationType.Linear);
        network.CreateHiddenLayers(3, Node.ActivationType.RElu);
        network.CreateOutputLayer(1, Node.ActivationType.Sigmoid);
        network.Randomize(0.5);
        
        // XOR dataset
        double[][] inputs = {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        
        double[][] outputs = {
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 }
        };
        
        // Run in parallel
        Parallel.For(0, 10, i => {
            Network localNet = NetworkUtilities.LoadFromFile(TestFileName);
            localNet.Optimize(inputs, outputs, 100, false);
            double loss = localNet.Loss(inputs, outputs);
            if (loss > 0.25) 
                throw new Exception($"Concurrent training failed (loss={loss})");
        });
        
        Console.WriteLine("Concurrency test passed");
    }
}*/