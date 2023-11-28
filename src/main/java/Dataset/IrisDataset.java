package Dataset;

import java.util.ArrayList;
import java.util.List;

public class IrisDataset {

    private List<IrisEntity> fullDataset;
    private List<IrisEntity> trainDataset;
    private List<IrisEntity> testDataset;

    public List<IrisEntity> getFullDataset() {
        return fullDataset;
    }

    public void setFullDataset(List<IrisEntity> fullDataset) {
        this.fullDataset = fullDataset;
    }

    public List<IrisEntity> getTrainDataset() {
        return trainDataset;
    }

    public void setTrainDataset(List<IrisEntity> trainDataset) {
        this.trainDataset = trainDataset;
    }

    public List<IrisEntity> getTestDataset() {
        return testDataset;
    }

    public void setTestDataset(List<IrisEntity> testDataset) {
        this.testDataset = testDataset;
    }

    public void InitTrainAndTestDataset (Float splitRatio){
        List<IrisEntity> temp = new ArrayList<>(this.fullDataset);
        int splitIndex = (int) (temp.size() * splitRatio);
        this.trainDataset = temp.subList(0, splitIndex);

        this.testDataset = temp;
        this.testDataset.removeAll(this.trainDataset);
    }
}
