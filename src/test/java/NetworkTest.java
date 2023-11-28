import Network.ElmanNetwork;

public class NetworkTest {
    public static void main(String[] args) {
        Integer inputSize = 4;
        Integer hiddenSize = 8;
        Integer outputSize = 3;

        ElmanNetwork elmanNetwork = new ElmanNetwork(inputSize, hiddenSize, outputSize);

        Float lambda = 0.001F;
        Float learningRate = 0.001F;
        Float momentum = 0.9F;
        Integer epochs = 100;

    }
}
