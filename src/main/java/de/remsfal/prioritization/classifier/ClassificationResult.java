package de.remsfal.prioritization.classifier;

public class ClassificationResult {

    private final String label;      // HIGH/MEDIUM/LOW
    private final double score;      // 0..1
    private final String modelVersion;

    public ClassificationResult(String label, double score, String modelVersion) {
        this.label = label;
        this.score = score;
        this.modelVersion = modelVersion;
    }

    public String getLabel() {
        return label;
    }

    public double getScore() {
        return score;
    }

    public String getModelVersion() {
        return modelVersion;
    }
}
