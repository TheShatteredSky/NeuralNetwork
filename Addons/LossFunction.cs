namespace NeuralNetwork.Addons;

internal static class LossFunction
{
    internal static double MSE(double x, double y) => (y - x) * (y - x);
    internal static double BinaryCrossEntropy(double x, double y) => -(y * Math.Log(x) + (1 - y) * Math.Log(1 - x));
    internal static double CategoricalCrossEntropy(double x, double y) => -(y * Math.Log(x));
}