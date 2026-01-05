package de.remsfal.prioritization.inference;

public class InferenceDtos {

    public record PredictRequest(String title, String description) {}

    public record PredictResponse(String priority, double score, String modelVersion) {}
}
