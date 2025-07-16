namespace NeuralNetwork.Core;

internal static class ActivationFunction
{
    internal static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    internal static double Tanh(double x) => Math.Tanh(x);
    internal static double RElu(double x) => x > 0 ? x : 0;
    internal static double LeakyRElu(double x) => x > 0 ? x : 0.01 * x;
    internal static double And(double x) => (x * x - x) / 2;
    internal static double Nand(double x) => (-x * x + x + 2) / 2;
    internal static double Or(double x) => (-x * x + 3 * x) / 2;
    internal static double Nor(double x) => (x * x - 3 * x + 2) / 2;
    internal static double Ex(double x) => -x * x + 2 * x;
    internal static double Nex(double x) => x * x - 2 * x + 1;
}