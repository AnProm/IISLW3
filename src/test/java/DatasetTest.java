import Dataset.DatasetUtil;
import Dataset.IrisEntity;

import java.util.List;

public class DatasetTest {
    public static void main(String[] args) {
        List<IrisEntity> dataset =  DatasetUtil.readDataset("C:\\Users\\user\\IdeaProjects\\IISLW3\\source\\Iris.csv", DatasetUtil.IRIS);
        System.out.println("test");
        DatasetUtil.shuffle(dataset);
        System.out.println("test2");
    }
}
