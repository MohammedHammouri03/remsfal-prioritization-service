package de.remsfal.prioritization.classifier;

public interface ClassifierStrategy {
    ClassificationResult predict(String title, String description);
}
