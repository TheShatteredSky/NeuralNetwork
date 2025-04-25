namespace NeuralNetwork;

public static class Functions
{
    public static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    public static double And(double x) => x * x / 4 + x / 2 - 1;
    public static double Nand(double x) => - x * x / 4 - x / 2 + 1;
    public static double Or(double x) => - x * x / 4 + x / 2 + 1;
    public static double Nor(double x) => x * x / 4 - x / 2 - 1;
    public static double Ex(double x) => - x * x / 2 + 1;
    public static double Nex(double x) => x * x / 2 - 1;
}