package Dataset;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DatasetUtil {
    public static final String IRIS = "IRIS";

    public static List<String[]> readDatasetFromFile(String path) {
        List<String[]> data = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(path))) {
            System.out.println("######[DATASET PRINT START]######");
            data = reader.readAll();
            // data содержит все строки CSV файла, каждая строка представлена как массив String

            // Пример вывода данных
            for (String[] row : data) {
                for (String cell : row) {
                    System.out.print(cell + "\t");
                }
                System.out.println();
            }
            System.out.println("######[DATASET PRINT END]######");
        } catch (IOException | CsvException e) {
            e.printStackTrace();
        }
        return data;
    }

    public static List<IrisEntity> readDataset(String path, String type) {
        List<IrisEntity> dataset = new ArrayList<>();
        List<String[]> rawData = readDatasetFromFile(path);

        for (String[] row : rawData) {
            if (Objects.equals(row[0], "Id")) {
                continue;
            }
            IrisEntity entity = new IrisEntity(Integer.parseInt(row[0]), Float.parseFloat(row[1]), Float.parseFloat(row[2]), Float.parseFloat(row[3]), Float.parseFloat(row[4]), irisCheckSpecies(row[5]));
            dataset.add(entity);
        }

        return dataset;
    }

    private static Integer irisCheckSpecies(String type) {
        int res = -1;
        if (Objects.equals(type, "Iris-setosa")) {
            res = 0;
        } else if (Objects.equals(type, "Iris-versicolor")) {
            res = 1;
        } else {
            res = 2;
        }
        return res;
    }

    public static void shuffle(List<IrisEntity> list) {
        int n = list.size();
        Random random = new Random();

        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);

            // Обмен местами элементов на позициях i и j
            IrisEntity temp = list.get(i);
            list.set(i, list.get(j));
            list.set(j, temp);
        }
    }
}
