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
    private double[][] prevHiddenToHiddenGradient;

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
        this.prevHiddenToHiddenGradient = initWeights(hiddenSize, hiddenSize);
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

    private double[] updateHiddenState(double[] input) { // Обновление состояния скрытого слоя и контекстного слоя
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

    private double[] feedForward(double[] input) {//Прмяой проход
        hiddenLayer = updateHiddenState(input);// обновить состояние скрытого слоя (вычисляет активации для скрытого слоя, учитывая входные данные и предыдущее состояние скрытого слоя)

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

    private void backPropagationWithMomentum(double[] input, double[] target, double lambda, double learningRate, double momentum) {//обратное распространение ошибки
        double[] output = feedForward(input);

        double[] outputErrors = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputErrors[i] = output[i] * (1 - output[i]) * (target[i] - output[i]);//вычисление ошибок для каждого выходного нейрона с использованием функции активации сигмоида и формулы ошибки для выходного слоя нейронной сети
        }

        double[][] hiddenToOutputGradients = new double[hiddenSize][outputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                hiddenToOutputGradients[i][j] = outputErrors[j] * hiddenLayer[i];//вычисление градиентов для скрытого слоя до выходного, перебор всех весов между скрытым слоем и выходным слоем для вычисления градиентов
            }
        }

        double[] hiddenErrors = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double error = 0;
            for (int j = 0; j < outputSize; j++) {
                error += outputErrors[j] * hiddenToOutputWeights[i][j];
            }
            hiddenErrors[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * error;//Вычисление ошибок для скрытого слоя
        }

        double[][] inputToHiddenGradients = new double[inputSize][hiddenSize];//Расчет градиентов весов между входным и скрытым слоями
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                inputToHiddenGradients[i][j] = hiddenErrors[j] * input[i];
            }
        }

        double[][] hiddenToHiddenGradients = new double[hiddenSize][hiddenSize];//Расчет градиентов скрытого слоя до скрытого слоя
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                hiddenToHiddenGradients[i][j] = hiddenErrors[j] * prevHiddenLayer[i];
            }
        }
        //Обновление всех весов
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double gradient = hiddenToOutputGradients[i][j] - lambda * hiddenToOutputWeights[i][j];
                double delta = learningRate * gradient + momentum * prevHiddenToOutputGradient[i][j];
                hiddenToOutputWeights[i][j] += delta;
                prevHiddenToOutputGradient[i][j] = delta;//Обновление весов между скрытым и выходным слоями, применение градиентного спуска с учетом момента для корректировки весов
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

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                double gradient = hiddenToHiddenGradients[i][j] - lambda * hiddenToHiddenWeights[i][j];
                double delta = learningRate * gradient + momentum * prevHiddenToHiddenGradient[i][j];
                hiddenToHiddenWeights[i][j] += delta;
                prevHiddenToHiddenGradient[i][j] = delta;
            }
        }
    }

    private double calculateLoss(double[] output, double[] target) {
        double loss = 0;
        for (int i = 0; i < outputSize; i++) {
            loss += Math.pow(target[i] - output[i], 2);// Использование квадратичной функции потерь (MSE)
        }
        return loss / outputSize;
    }

    public double[] train(List<double[]> trainInputs, List<double[]> trainTargets, double lambda, double learningRate, double momentum, int epochs) {
        double[] errors = new double[epochs];
        System.out.println("######[START TRAIN FOR "+epochs+" EPOCHS]######");
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
        System.out.println("######[END TRAIN FOR "+epochs+" EPOCHS]######");
        return errors;
    }
/*
    public double test(List<double[]> testInputs, List<double[]> testTargets) {
        int correctPredictions = 0;

        for (int i = 0; i < testInputs.size(); i++) {
            double[] output = feedForward(testInputs.get(i));
            int predictedClass = getMaxIndex(output);
            int actualClass = getMaxIndex(testTargets.get(i));

            if (predictedClass == actualClass) {
                correctPredictions++;
            }
        }

        double accuracy = ((double) correctPredictions / testInputs.size()) * 100;
        System.out.printf("Test Accuracy: %.5f%%\n", accuracy);
        return accuracy;
    }
*/
    public double[] test(List<double[]> testInputs, List<double[]> testTargets) {
        double[] losses = new double[testInputs.size()];

        for (int i = 0; i < testInputs.size(); i++) {
            double[] output = feedForward(testInputs.get(i));
            losses[i] = calculateLoss(output, testTargets.get(i));
        }

        //System.out.println("Test Losses:");
        for (double loss : losses) {
            //System.out.printf("%.5f\n", loss);
        }

        return losses;
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


        int inputSize = 4;
        int hiddenSize = 16;
        int outputSize = 3;

        ElmanNetwork elmanNet = new ElmanNetwork(inputSize, hiddenSize, outputSize);
        //4 - ok, 5 - ok, 6 - ok ,7 - ok

        double lambda = 0.001;
        double learningRate = 0.01; //коэффициент скорости обучения, который определяет величину изменения весов на каждой итерации
        double momentum = 0.9; //коэффициент, который определяет, как сильно предыдущее изменение весов должно влиять на текущее изменение
        int epochs = 30;

        //п. 4.1: Погрешность обучения от коэффицента обучения
        double[] learningRates = {0.01, 0.05, 0.1, 0.2, 0.3};
        for (double learningR : learningRates) {
            System.out.println("######[LEARNING RATE: "+learningR+"]######");
            elmanNet.train(trainInputs, trainTargets, lambda, learningR, momentum, epochs);
        }

        //elmanNet.train(trainInputs, trainTargets, lambda, learningRate, momentum, epochs);


        //п. 5.1: Погрешность обучения от числа нейронов скрытого слоя
        //п. 7.1: Погрешность обучения от числа эпох
        // Значения числа нейронов скрытого слоя для анализа зависимости погрешности обучения
        int[] hiddenLayerSizes = {4, 8, 12, 16, 256};
        int[] epochsValues = {50, 100, 150, 200};

        DefaultXYDataset errorDataset = new DefaultXYDataset();

        for (int hiddenS : hiddenLayerSizes) {
            double[][] data = new double[2][epochsValues.length];
            for (int i = 0; i < epochsValues.length; i++) {
                elmanNet = new ElmanNetwork(inputSize, hiddenS, outputSize);
                System.out.println("############[START TRAIN FOR "+hiddenS+" NEURONS]############");
                double[] errors = elmanNet.train(trainInputs, trainTargets, lambda, learningRate, momentum, epochsValues[i]);
                data[0][i] = epochsValues[i];
                data[1][i] = errors[epochsValues[i] - 1]; // берем ошибку на последней эпохе
            }
            errorDataset.addSeries("Hidden Layer Size: " + hiddenS, data);
        }

        JFreeChart errorChart = createChart("Error vs Epochs (train)", "Epochs", "Error", errorDataset);

        displayChart(errorChart);//РАСКОММЕНТИРОВАТЬ

        //п. 5.2/7.2: Погрешность классификации от числа нейронов скрытого слоя и числа эпох (КЛАССИФИКАЦИЯ)
        DefaultXYDataset classificationErrorDataset = new DefaultXYDataset();

        //#########################################################
        for (int hiddenS : hiddenLayerSizes) {
            double[][] data = new double[2][epochsValues.length];
            for (int i = 0; i < epochsValues.length; i++) {
                elmanNet = new ElmanNetwork(inputSize, hiddenS, outputSize);
                double[] losses = elmanNet.test(testInputs, testTargets);

                double sumLosses = 0;
                for (double loss : losses) {
                    sumLosses += loss;
                }
                double avgLoss = sumLosses / testInputs.size();

                data[0][i] = epochsValues[i];
                data[1][i] = avgLoss; // классификационная погрешность (среднее значение потерь)

                System.out.printf("Hidden Neurons: %d, Epochs: %d, Classification Loss: %.5f\n",
                        hiddenS, epochsValues[i], avgLoss);
            }
            classificationErrorDataset.addSeries("Hidden Layer Size: " + hiddenS, data);
        }

        JFreeChart classificationErrorChart = createChart("Classification Loss vs Epochs and Hidden Neurons (test)",
                "Epochs", "Classification Loss", classificationErrorDataset);

        displayChart(classificationErrorChart);
        //####################################################

        //п. 6 Зависимость погрешности классификации от объема обучающей выборки
        System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!PRINT DATASET SIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        DefaultXYDataset classificationErrorDataset2 = new DefaultXYDataset();

        int[] trainingSetSizes = {10, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120};

        double[][] data = new double[2][trainingSetSizes.length];

        for (int i = 0; i < trainingSetSizes.length; i++) {
            int trainSize2 = trainingSetSizes[i];

            List<double[]> currentTrainInputs = trainInputs.subList(0, trainSize2);
            List<double[]> currentTrainTargets = trainTargets.subList(0, trainSize2);

            ElmanNetwork elmanNet2 = new ElmanNetwork(inputSize, hiddenSize, outputSize);
            elmanNet2.train(currentTrainInputs, currentTrainTargets, lambda, learningRate, momentum, epochs);
            double[] losses = elmanNet2.test(testInputs, testTargets);

            double sumLosses = 0;
            for (double loss : losses) {
                sumLosses += loss;
            }
            double avgLoss = sumLosses / testInputs.size();

            double classificationError = avgLoss; // классификационная погрешность (среднее значение потерь)
            data[0][i] = trainSize2;
            data[1][i] = classificationError;

            System.out.printf("Training Set Size: %d, Classification Error: %.5f\n",
                    trainSize2, classificationError);
        }

        classificationErrorDataset2.addSeries("Classification Error", data);

        JFreeChart classificationErrorChart2 = createChart("Classification Error vs Training Set Size (test)",
                "Training Set Size", "Classification Error", classificationErrorDataset2);

        displayChart(classificationErrorChart2);

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
