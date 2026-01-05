package de.remsfal.prioritization.classifier;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "remsfal.classifier.mode", havingValue = "stub", matchIfMissing = true)
public class StubKeywordClassifier implements ClassifierStrategy {

    @Override
    public ClassificationResult predict(String text) {
        String lower = text == null ? "" : text.toLowerCase();

        // simple Heuristik nur für Pipeline-Test
        if (lower.contains("wasser") || lower.contains("brand") || lower.contains("notfall") || lower.contains("strom")) {
            return new ClassificationResult("HIGH", 0.90, "stub-keyword-v1");
        }
        if (lower.contains("heizung") || lower.contains("defekt") || lower.contains("kaputt")) {
            return new ClassificationResult("MEDIUM", 0.75, "stub-keyword-v1");
        }
        return new ClassificationResult("LOW", 0.60, "stub-keyword-v1");
    }
}
