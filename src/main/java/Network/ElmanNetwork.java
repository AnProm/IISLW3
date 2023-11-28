package Network;

import java.util.Random;

public class ElmanNetwork {

    private Integer inputLayerSize;
    private Integer hiddenLayerSize;
    private Integer outputLayerSize;

    private Float [][] inputToHiddenWeights;
    private Float [][] hiddenToHiddenWeights;
    private Float [][] hiddenToOutputWeights;

    private Float [] hiddenLayer;
    private Float [] prevHiddenLayer;

    private Float [][] prevInputToHiddenGradient;
    private Float [][] prevHiddenToOutputGradient;

    public ElmanNetwork(Integer inputLayerSize, Integer hiddenLayerSize, Integer outputLayerSize) {
        this.inputLayerSize = inputLayerSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.outputLayerSize = outputLayerSize;

        this.inputToHiddenWeights = initializeRandomMatrix(inputLayerSize, hiddenLayerSize);
        this.hiddenToHiddenWeights = initializeRandomMatrix(hiddenLayerSize, hiddenLayerSize);
        this.hiddenToOutputWeights = initializeRandomMatrix(hiddenLayerSize, outputLayerSize);

        this.hiddenLayer = initializeFloatArray(hiddenLayerSize);
        this.prevHiddenLayer = initializeFloatArray(hiddenLayerSize);

        this.prevInputToHiddenGradient = initializeRandomMatrix(inputLayerSize, hiddenLayerSize);
        this.prevHiddenToOutputGradient = initializeRandomMatrix(hiddenLayerSize, outputLayerSize);
    }

    private static Float sigmoid(Float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    private static Float[][] initializeRandomMatrix(int rows, int cols) {
        Float[][] matrix = new Float[rows][cols];
        Random random = new Random();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextFloat(); // Генерация случайного числа от 0 до 1
            }
        }
        return matrix;
    }

    private static Float[] initializeFloatArray(int length) {
        Float[] floatArray = new Float[length];
        for (int i = 0; i < length; i++) {
            floatArray[i] = 0.0f; // Заполнение массива нулями
        }
        return floatArray;
    }
}
