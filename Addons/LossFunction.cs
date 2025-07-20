namespace NeuralNetwork.Addons;

internal static class LossFunction
{
    private static double _epsilon = 1e-16;
    internal static double MSE(double x, double y) => (y - x) * (y - x);
    internal static double BinaryCrossEntropy(double x, double y) => -(y * Math.Log(x + _epsilon) + (1 - y) * Math.Log(1 - x + _epsilon));
    //I honestly don't really understand why this formula works, but I've checked everywhere, and it's correct so ¯\_(ツ)_/¯
    internal static double CategoricalCrossEntropy(double x, double y) => -(y * Math.Log(x + _epsilon));
}