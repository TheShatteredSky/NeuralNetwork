namespace NeuralNetwork;

public static class Functions
{
    public static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
}