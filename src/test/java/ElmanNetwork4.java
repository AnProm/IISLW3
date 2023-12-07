import Dataset.DatasetUtil;
import Dataset.IrisEntity;

import javax.swing.*;
import org.math.plot.Plot2DPanel;

import java.io.*;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class ElmanNetwork4 {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[][] contextWeights;
    private double[] hiddenLayer;
    private double[] contextLayer;

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

    public ElmanNetwork4() {
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

    public double train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        double globalMeanLoss = 0.0;//надстройка
        double globalAccuracy = 0.0;//надстройка

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

            globalAccuracy += accuracy;//надстройка
            globalMeanLoss += meanLoss;//надстройка

            System.out.println("Epoch: " + (epoch + 1) + " - Loss: " + meanLoss + " - Accuracy: " + accuracy);
        }

        globalMeanLoss /= epochs;//надстройка (на выбор)
        globalAccuracy /= epochs;//надстройка

        return globalMeanLoss;

    }

    public double test(double[][] inputs, double[][] targets) {
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

        return meanLoss;//на выбор
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

    public void saveToFile() {
        String filename = "C:\\Users\\user\\Desktop\\NNELMAN.txt";
        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(filename))) {
            // Save weights and context layer
            outputStream.writeObject(this.weightsInputHidden);
            outputStream.writeObject(this.weightsHiddenOutput);
            outputStream.writeObject(this.contextWeights);
            outputStream.writeObject(this.hiddenLayer);
            outputStream.writeObject(this.contextLayer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void loadFromFile() {
        String filename = "C:\\Users\\user\\Desktop\\NNELMAN.txt";
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(filename))) {
            // Load weights and context layer
            this.weightsInputHidden = (double[][]) inputStream.readObject();
            this.weightsHiddenOutput = (double[][]) inputStream.readObject();
            this.contextWeights = (double[][]) inputStream.readObject();
            this.hiddenLayer = (double[])  inputStream.readObject();
            this.contextLayer = (double[]) inputStream.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
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

        //Для обучения
        double[][] inputsArray = convertListToArray(trainInputs);
        double[][] targetsArray = convertListToArray(trainTargets);
        //Для тестирования
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

        //#######################P.4############################
        double[] learningRates = {0.001, 0.01, 0.1, 0.5, 1.0};
        double[] lossValues = new double[learningRates.length];

        ElmanNetwork4 network2 = new ElmanNetwork4(inputSize, hiddenSize, outputSize);//СОБЛЮДАЕМ ВЕСА
        network2.saveToFile();
        for (int i = 0; i < learningRates.length; i++) {
            network2.loadFromFile();
            lossValues[i] = network2.train(inputsArray, targetsArray, epochs, learningRates[i]);
        }

        plotMeanLossVsLearningRate(learningRates, lossValues);
        //#######################P.4#END########################
        //#######################P.5############################
        int[] hiddenSizes = {5, 10, 15, 20, 25}; // Различные значения числа нейронов в скрытом слое

        double[][] res = new ElmanNetwork4().trainWithDifferentHiddenSizes(hiddenSizes, inputsArray, targetsArray,testInputsArray, testTargetsArray, epochs, learningRate);
        double[] trainLosses = res[0];
        double[] testLosses = res[1];

        new ElmanNetwork4().plotLossVsHiddenNeurons(hiddenSizes, trainLosses, testLosses);
        //#######################P.5#END########################
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


    public static void plotMeanLossVsLearningRate(double[] learningRates, double[] lossValues) {
        Plot2DPanel plot = new Plot2DPanel("LEARNING RATE");
        plot.addLinePlot("Mean Loss vs Learning Rate", learningRates, lossValues);

        JFrame frame = new JFrame("Mean Loss vs Learning Rate");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }


    //tyajelo...

    public void plotLossVsHiddenNeurons(int[] hiddenSizes, double[] trainLosses, double[] testLosses) {
        Plot2DPanel plot = new Plot2DPanel();
        plot.addLinePlot("Train Loss vs Hidden Neurons", intArrayToDoubleArray(hiddenSizes), trainLosses);
        plot.addLinePlot("Test Loss vs Hidden Neurons", intArrayToDoubleArray(hiddenSizes), testLosses);

        JFrame frame = new JFrame("Loss vs Hidden Neurons");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }

    public static double[] intArrayToDoubleArray(int[] intArray) {
        double[] doubleArray = new double[intArray.length];

        for (int i = 0; i < intArray.length; i++) {
            doubleArray[i] = (double) intArray[i];
        }

        return doubleArray;
    }

    public double[][] trainWithDifferentHiddenSizes(int[] hiddenSizes, double[][] inputs, double[][] targets,double[][] inputsTEST, double[][] targetsTEST, int epochs, double learningRate) {
        double[] trainLosses = new double[hiddenSizes.length];
        double[] testLosses = new double[hiddenSizes.length];
        int inputSize = 4;
        int outputSize = 3;
        learningRate = 0.1;

        for (int i = 0; i < hiddenSizes.length; i++) {
            int hiddenSize = hiddenSizes[i];
            ElmanNetwork4 network = new ElmanNetwork4(inputSize, hiddenSize, outputSize);
            trainLosses[i] = network.train(inputs, targets, epochs, learningRate);
            testLosses[i] = network.test(inputsTEST, targetsTEST);
        }

        double[][] outp = new double[2][hiddenSizes.length];
        outp[0] = trainLosses;
        outp[1] = testLosses;

        return outp;
    }
}
