import Dataset.DatasetUtil;
import Dataset.IrisEntity;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultXYDataset;

import javax.swing.*;
import java.awt.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ElmanNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;

    private double[][] inputToHiddenWeights;
    private double[][] hiddenToHiddenWeights;
    private double[][] hiddenToOutputWeights;

    private double[] hiddenLayer;
    private double[] prevHiddenLayer;

    private double[][] prevInputToHiddenGradient;
    private double[][] prevHiddenToOutputGradient;

    public ElmanNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.inputToHiddenWeights = initWeights(inputSize, hiddenSize);
        this.hiddenToHiddenWeights = initWeights(hiddenSize, hiddenSize);
        this.hiddenToOutputWeights = initWeights(hiddenSize, outputSize);

        this.hiddenLayer = new double[hiddenSize];
        this.prevHiddenLayer = new double[hiddenSize];

        this.prevInputToHiddenGradient = initWeights(inputSize, hiddenSize);
        this.prevHiddenToOutputGradient = initWeights(hiddenSize, outputSize);
    }

    private double[][] initWeights(int rows, int cols) {
        double[][] weights = new double[rows][cols];
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = random.nextDouble() - 0.5;
            }
        }
        return weights;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double[] updateHiddenState(double[] input) {
        double[] currentHiddenLayer = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < inputSize; j++) {
                sum += input[j] * inputToHiddenWeights[j][i];
            }
            for (int j = 0; j < hiddenSize; j++) {
                sum += prevHiddenLayer[j] * hiddenToHiddenWeights[j][i];
            }
            currentHiddenLayer[i] = sigmoid(sum);
        }

        prevHiddenLayer = currentHiddenLayer;
        return currentHiddenLayer;
    }

    private double[] feedForward(double[] input) {
        hiddenLayer = updateHiddenState(input);

        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayer[j] * hiddenToOutputWeights[j][i];
            }
            output[i] = sigmoid(sum);
        }

        return output;
    }

    private void backPropagationWithMomentum(double[] input, double[] target, double lambda, double learningRate, double momentum) {
        double[] output = feedForward(input);

        double[] outputErrors = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputErrors[i] = output[i] * (1 - output[i]) * (target[i] - output[i]);
        }

        double[][] hiddenToOutputGradients = new double[hiddenSize][outputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                hiddenToOutputGradients[i][j] = outputErrors[j] * hiddenLayer[i];
            }
        }

        double[] hiddenErrors = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double error = 0;
            for (int j = 0; j < outputSize; j++) {
                error += outputErrors[j] * hiddenToOutputWeights[i][j];
            }
            hiddenErrors[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * error;
        }

        double[][] inputToHiddenGradients = new double[inputSize][hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                inputToHiddenGradients[i][j] = hiddenErrors[j] * input[i];
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double gradient = hiddenToOutputGradients[i][j] - lambda * hiddenToOutputWeights[i][j];
                double delta = learningRate * gradient + momentum * prevHiddenToOutputGradient[i][j];
                hiddenToOutputWeights[i][j] += delta;
                prevHiddenToOutputGradient[i][j] = delta;
            }
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                double gradient = inputToHiddenGradients[i][j] - lambda * inputToHiddenWeights[i][j];
                double delta = learningRate * gradient + momentum * prevInputToHiddenGradient[i][j];
                inputToHiddenWeights[i][j] += delta;
                prevInputToHiddenGradient[i][j] = delta;
            }
        }
    }

    private double calculateLoss(double[] output, double[] target) {
        double loss = 0;
        for (int i = 0; i < outputSize; i++) {
            loss += Math.pow(target[i] - output[i], 2);
        }
        return loss / outputSize;
    }

    public double[] train(List<double[]> trainInputs, List<double[]> trainTargets, double lambda, double learningRate, double momentum, int epochs) {
        double[] errors = new double[epochs];
        System.out.println("######[START TRAIN FOR "+epochs+"EPOCHS]######");
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int correctPredictions = 0;

            for (int i = 0; i < trainInputs.size(); i++) {
                backPropagationWithMomentum(trainInputs.get(i), trainTargets.get(i), lambda, learningRate, momentum);

                double[] output = feedForward(trainInputs.get(i));
                totalLoss += calculateLoss(output, trainTargets.get(i));

                int predictedClass = getMaxIndex(output);
                int actualClass = getMaxIndex(trainTargets.get(i));

                if (predictedClass == actualClass) {
                    correctPredictions++;
                }
            }

            double epochLoss = totalLoss / trainInputs.size();
            errors[epoch] = epochLoss;
            double epochAccuracy = ((double) correctPredictions / trainInputs.size()) * 100;

            System.out.printf("Epoch %d: Loss = %.5f, Accuracy = %.5f%%\n", epoch + 1, epochLoss, epochAccuracy);
        }
        System.out.println("######[END TRAIN FOR "+epochs+"EPOCHS]######");
        return errors;
    }
/*
    private static double[] trainAndGetErrors(List<double[]> trainInputs, List<double[]> trainTargets, List<IrisEntity> data,
                                              double lambda, double learningRate, double momentum, int epochs) {
        double[] errors = new double[epochs];
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            for (IrisEntity item : data) {
                double[] input = {item.sepalLengthCm, item.sepalWidthCm, item.petalLengthCm, item.petalWidthCm};
                double[] target = convertToTarget(item.species);

                elmanNet.backPropagationWithMomentum(input, target, lambda, learningRate, momentum);

                double[] output = elmanNet.feedForward(input);
                totalLoss += elmanNet.calculateLoss(output, target);
            }

            errors[epoch] = totalLoss / data.size();
        }
        return errors;
    }*/

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


        int inputSize = 4;
        int hiddenSize = 8;
        int outputSize = 3;

        ElmanNetwork elmanNet = new ElmanNetwork(inputSize, hiddenSize, outputSize);

        double lambda = 0.001;
        double learningRate = 0.01;
        double momentum = 0.9;
        int epochs = 100;

        elmanNet.train(trainInputs, trainTargets, lambda, learningRate, momentum, epochs);

        // Пример использования для построения графиков
        // Значения числа нейронов скрытого слоя для анализа зависимости погрешности обучения
        int[] hiddenLayerSizes = {4, 8, 12, 16, 256};
        int[] epochsValues = {50, 100, 150, 200};

        DefaultXYDataset errorDataset = new DefaultXYDataset();

        for (int hiddenSize1 : hiddenLayerSizes) {
            double[][] data = new double[2][epochsValues.length];
            for (int i = 0; i < epochsValues.length; i++) {
                ElmanNetwork elmanNet1 = new ElmanNetwork(inputSize, hiddenSize1, outputSize);
                System.out.println("############[START TRAIN FOR "+hiddenSize1+" NEURONS]############");
                double[] errors = elmanNet1.train(trainInputs, trainTargets, lambda, learningRate, momentum, epochsValues[i]);
                data[0][i] = epochsValues[i];
                data[1][i] = errors[epochsValues[i] - 1]; // берем ошибку на последней эпохе
            }
            errorDataset.addSeries("Hidden Layer Size: " + hiddenSize1, data);
        }

        JFreeChart errorChart = createChart("Error vs Epochs", "Epochs", "Error", errorDataset);

        displayChart(errorChart);
    }


    private static double[] convertToTarget(int species) {
        // Метод для преобразования значения класса в вектор целей
        double[] target = new double[3];
        target[species] = 1;
        return target;
    }

    private static int getMaxIndex(double[] array) {
        int maxIndex = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static void addDataToDataset(DefaultXYDataset dataset, String seriesName, int[] xData, double[] yData) {
        double[][] data = new double[2][xData.length];
        for (int i = 0; i < xData.length; i++) {
            data[0][i] = xData[i];
            data[1][i] = yData[i];
        }
        dataset.addSeries(seriesName, data);
    }

    private static JFreeChart createChart(String title, String xAxisLabel, String yAxisLabel, DefaultXYDataset dataset) {
        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                xAxisLabel,
                yAxisLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        return chart;
    }

    private static void displayChart(JFreeChart chart) {
        EventQueue.invokeLater(() -> {
            JFrame frame = new JFrame("Chart");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.getContentPane().add(new ChartPanel(chart));
            frame.pack();
            frame.setVisible(true);
        });
    }
}
