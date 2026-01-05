package de.remsfal.prioritization.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import de.remsfal.prioritization.classifier.ClassificationResult;
import de.remsfal.prioritization.classifier.ClassifierStrategy;
import de.remsfal.prioritization.config.KafkaTopicsConfig;
import de.remsfal.prioritization.model.IssueCreatedEvent;
import de.remsfal.prioritization.model.IssuePriorityResultEvent;
import de.remsfal.prioritization.publisher.PriorityResultPublisher;
import de.remsfal.prioritization.validation.IssueCreatedValidator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.time.Instant;

@Component
public class IssueCreatedConsumer {

    private static final Logger log = LoggerFactory.getLogger(IssueCreatedConsumer.class);

    private final ObjectMapper objectMapper;
    private final KafkaTopicsConfig topics;
    private final IssueCreatedValidator validator;
    private final ClassifierStrategy classifier;
    private final PriorityResultPublisher publisher;

    public IssueCreatedConsumer(
            ObjectMapper objectMapper,
            KafkaTopicsConfig topics,
            IssueCreatedValidator validator,
            ClassifierStrategy classifier,
            PriorityResultPublisher publisher
    ) {
        this.objectMapper = objectMapper;
        this.topics = topics;
        this.validator = validator;
        this.classifier = classifier;
        this.publisher = publisher;
    }

    @KafkaListener(topics = "${remsfal.kafka.topics.in}", groupId = "${spring.kafka.consumer.group-id}")

    public void consume(String payload) {
        try {
            IssueCreatedEvent event = objectMapper.readValue(payload, IssueCreatedEvent.class);


            validator.validate(event);


            String title = event.getTitle() == null ? "" : event.getTitle().trim();
            String desc  = event.getDescription() == null ? "" : event.getDescription().trim();
            String text  = (title + "\n" + desc).trim();

            ClassificationResult result = classifier.predict(text);



            IssuePriorityResultEvent out = new IssuePriorityResultEvent();
            out.setIssueId(event.getIssueId());
            out.setProjectId(event.getProjectId());
            out.setPriority(result.getLabel());
            out.setPriorityScore(result.getScore());
            out.setPriorityModel(result.getModelVersion());
            out.setPriorityTimestamp(Instant.now());


            publisher.publish(out);

        } catch (Exception ex) {

            log.warn("Failed to process incoming issue event. payload={}", payload, ex);
        }
    }
}

