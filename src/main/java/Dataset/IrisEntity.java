package Dataset;

public class IrisEntity {

    public Integer id;
    public Float sepalLengthCm;
    public Float sepalWidthCm;
    public Float petalLengthCm;
    public Float petalWidthCm;
    public Integer species;

    public IrisEntity(Integer id, Float sepalLengthCm, Float sepalWidthCm, Float petalLengthCm, Float petalWidthCm, Integer species) {
        this.id = id;
        this.sepalLengthCm = sepalLengthCm;
        this.sepalWidthCm = sepalWidthCm;
        this.petalLengthCm = petalLengthCm;
        this.petalWidthCm = petalWidthCm;
        this.species = species;
    }
}
