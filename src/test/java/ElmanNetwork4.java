import Dataset.DatasetUtil;
import Dataset.IrisEntity;

import java.util.Arrays;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ElmanNetwork4 {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final double[][] weightsInputHidden;
    private final double[][] weightsHiddenOutput;
    private final double[][] contextWeights;
    private final double[] hiddenLayer;
    private final double[] contextLayer;

    private double learningRate = 0.01; // Попробуйте уменьшить learning rate


    public ElmanNetwork4(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.weightsInputHidden = new double[inputSize][hiddenSize];
        this.weightsHiddenOutput = new double[hiddenSize][outputSize];
        this.contextWeights = new double[hiddenSize][hiddenSize];

        this.hiddenLayer = new double[hiddenSize];
        this.contextLayer = new double[hiddenSize];

        initializeWeights();
    }

    private void initializeWeights() {
        // Инициализируем веса в более узком диапазоне (-0.1 до 0.1)
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = Math.random() * 0.2 - 0.1;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput[i][j] = Math.random() * 0.2 - 0.1;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                contextWeights[i][j] = Math.random() * 0.2 - 0.1;
            }
        }
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double tanh(double x) {
        return Math.tanh(x);
    }

    private double[] forwardPropagation(double[] input) {
        // Forward pass through the network
        // Compute hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += input[j] * weightsInputHidden[j][i];
            }
            for (int j = 0; j < hiddenSize; j++) {
                sum += contextLayer[j] * contextWeights[j][i];
            }
            //hiddenLayer[i] = tanh(sum);
            hiddenLayer[i] = sigmoid(sum);
        }

        // Compute output layer
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayer[j] * weightsHiddenOutput[j][i];
            }
            output[i] = sigmoid(sum);
        }

        // Update context layer
        System.arraycopy(hiddenLayer, 0, contextLayer, 0, hiddenSize);

        return output;
    }

    private void backwardPropagation(double[] input, double[] target, double learningRate) {
        // Backward pass through the network to update weights
        double[] output = forwardPropagation(input);

        // Compute output layer errors
        double[] outputErrors = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputErrors[i] = (target[i] - output[i]) * output[i] * (1 - output[i]);
        }

        // Update hidden to output weights
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput[i][j] += learningRate * outputErrors[j] * hiddenLayer[i];
            }
        }

        // Compute hidden layer errors
        double[] hiddenErrors = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double error = 0.0;
            for (int j = 0; j < outputSize; j++) {
                error += outputErrors[j] * weightsHiddenOutput[i][j];
            }
            hiddenErrors[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * error;
            //hiddenErrors[i] = (1 - hiddenLayer[i] * hiddenLayer[i]) * error;
        }

        // Update input to hidden weights and context weights
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * input[i];
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                contextWeights[i][j] += learningRate * hiddenErrors[j] * contextLayer[i];
            }
        }
    }

    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            int correctPredictions = 0;

            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                double[] target = targets[i];

                double[] output = forwardPropagation(input);

                // Calculate loss (mean squared error)
                double loss = 0.0;
                for (int j = 0; j < outputSize; j++) {
                    loss += Math.pow(target[j] - output[j], 2);
                }
                totalLoss += loss / outputSize;

                // Update weights via backpropagation
                backwardPropagation(input, target, learningRate);

                // Check accuracy
                boolean correct = true;
                for (int j = 0; j < outputSize; j++) {
                    if (Math.round(output[j]) != target[j]) {
                        correct = false;
                        break;
                    }
                }
                if (correct) {
                    correctPredictions++;
                }
            }

            double accuracy = (double) correctPredictions / inputs.length;
            double meanLoss = totalLoss / inputs.length;

            System.out.println("Epoch: " + (epoch + 1) + " - Loss: " + meanLoss + " - Accuracy: " + accuracy);
        }
    }

    public void test(double[][] inputs, double[][] targets) {
        double totalLoss = 0.0;
        int correctPredictions = 0;

        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] target = targets[i];

            double[] output = forwardPropagation(input);

            // Calculate loss (mean squared error)
            double loss = 0.0;
            for (int j = 0; j < outputSize; j++) {
                loss += Math.pow(target[j] - output[j], 2);
            }
            totalLoss += loss / outputSize;

            // Check accuracy
            boolean correct = true;
            for (int j = 0; j < outputSize; j++) {
                if (Math.round(output[j]) != target[j]) {
                    correct = false;
                    break;
                }
            }
            if (correct) {
                correctPredictions++;
            }
        }

        double accuracy = (double) correctPredictions / inputs.length;
        double meanLoss = totalLoss / inputs.length;

        System.out.println("Test Results - Loss: " + meanLoss + " - Accuracy: " + accuracy);
    }

    private double[] roundOutput(double[] output) {
        double[] rounded = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            rounded[i] = Math.round(output[i]);
        }
        return rounded;
    }

    private static double[] convertToTarget(int species) {
        // Метод для преобразования значения класса в вектор целей
        double[] target = new double[3];
        target[species] = 1;
        return target;
    }

    public static void main(String[] args) {
        List<IrisEntity> irisData = DatasetUtil.readDataset("C:\\Users\\user\\IdeaProjects\\IISLW3\\source\\Iris.csv", DatasetUtil.IRIS);
        DatasetUtil.shuffle(irisData);

        List<double[]> inputs = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();

        for (IrisEntity item : irisData) {
            double[] input = {
                    item.sepalLengthCm, item.sepalWidthCm,
                    item.petalLengthCm, item.petalWidthCm
            };
            inputs.add(input);

            double[] target = convertToTarget(item.species);
            targets.add(target);
        }

        int trainSize = (int) (0.8 * inputs.size());
        List<double[]> trainInputs = inputs.subList(0, trainSize);
        List<double[]> trainTargets = targets.subList(0, trainSize);
        List<double[]> testInputs = inputs.subList(trainSize, inputs.size());
        List<double[]> testTargets = targets.subList(trainSize, inputs.size());

        double[][] inputsArray = convertListToArray(trainInputs);
        double[][] targetsArray = convertListToArray(trainTargets);

        double[][] testInputsArray = convertListToArray(testInputs);
        double[][] testTargetsArray = convertListToArray(testTargets);

        // Create and train the network
        int inputSize = 4; // 4 features in Iris dataset
        int hiddenSize = 10;
        int outputSize = 3; // 3 Iris types

        int epochs = 5000;
        double learningRate = 0.01;

        ElmanNetwork4 network = new ElmanNetwork4(inputSize, hiddenSize, outputSize);
        network.train(inputsArray, targetsArray, epochs, learningRate);

        // Test the network
        network.test(testInputsArray, testTargetsArray);
    }

    public static double[][] convertListToArray(List<double[]> list) {
        int size = list.size();
        int innerSize = list.get(0).length;

        double[][] array = new double[size][innerSize];

        for (int i = 0; i < size; i++) {
            array[i] = list.get(i);
        }

        return array;
    }
}
