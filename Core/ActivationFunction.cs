namespace NeuralNetwork.Core;

internal static class ActivationFunction
{
    internal static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    internal static double TanH(double x) => Math.Tanh(x);
    internal static double ReLU(double x) => x > 0 ? x : 0;
    internal static double LeakyReLU(double x) => x > 0 ? x : 0.01 * x;
    internal static double AND(double x) => (x * x - x) / 2;
    internal static double NAND(double x) => (-x * x + x + 2) / 2;
    internal static double OR(double x) => (-x * x + 3 * x) / 2;
    internal static double NOR(double x) => (x * x - 3 * x + 2) / 2;
    internal static double EX(double x) => -x * x + 2 * x;
    internal static double NEX(double x) => x * x - 2 * x + 1;
}