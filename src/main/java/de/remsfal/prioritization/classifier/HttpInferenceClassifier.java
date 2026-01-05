package de.remsfal.prioritization.classifier;

import de.remsfal.prioritization.config.InferenceConfig;
import de.remsfal.prioritization.inference.InferenceDtos.PredictRequest;
import de.remsfal.prioritization.inference.InferenceDtos.PredictResponse;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;

@Component
@ConditionalOnProperty(name = "remsfal.classifier.mode", havingValue = "inference")
public class HttpInferenceClassifier implements ClassifierStrategy {

    private final WebClient webClient;
    private final InferenceConfig cfg;

    public HttpInferenceClassifier(WebClient inferenceWebClient, InferenceConfig cfg) {
        this.webClient = inferenceWebClient;
        this.cfg = cfg;
    }

    @Override
    public ClassificationResult predict(String title, String description) {
        String safeTitle = (title == null || title.isBlank()) ? "Issue" : title.trim();
        String safeDesc  = (description == null) ? "" : description.trim();

        String endpoint = mapProviderToEndpoint(cfg.getProvider());

        PredictResponse resp = webClient.post()
                .uri(endpoint)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(new PredictRequest(safeTitle, safeDesc))
                .retrieve()
                .bodyToMono(PredictResponse.class)
                .timeout(Duration.ofMillis(cfg.getTimeoutMs()))
                .block();

        if (resp == null || resp.priority() == null) {
            return new ClassificationResult("MEDIUM", 0.0, "inference-null");
        }

        return new ClassificationResult(resp.priority(), resp.score(), resp.modelVersion());
    }


    private String mapProviderToEndpoint(String provider) {
        if (provider == null) return "/predict/baseline";
        return switch (provider.toLowerCase()) {
            case "tfidf", "baseline" -> "/predict/baseline";
            case "xlmr" -> "/predict/xlmr";
            case "openai" -> "/predict/openai";
            default -> "/predict/baseline";
        };
    }
}
