package de.remsfal.prioritization.publisher;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.remsfal.prioritization.config.KafkaTopicsConfig;
import de.remsfal.prioritization.model.IssuePriorityResultEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

@Component
public class PriorityResultPublisher {

    private static final Logger log = LoggerFactory.getLogger(PriorityResultPublisher.class);

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;
    private final KafkaTopicsConfig topics;

    public PriorityResultPublisher(
            KafkaTemplate<String, String> kafkaTemplate,
            ObjectMapper objectMapper,
            KafkaTopicsConfig topics
    ) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
        this.topics = topics;
    }

    public void publish(IssuePriorityResultEvent event) {
        try {
            String payload = objectMapper.writeValueAsString(event);
            String key = event.getIssueId().toString();

            kafkaTemplate.send(topics.getOut(), key, payload);
            log.info("Published priority result issueId={} priority={} score={}",
                    event.getIssueId(), event.getPriority(), event.getPriorityScore());
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize IssuePriorityResultEvent", e);
        }
    }
}
