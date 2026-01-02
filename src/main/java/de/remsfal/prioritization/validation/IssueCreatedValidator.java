package de.remsfal.prioritization.validation;

import de.remsfal.prioritization.model.IssueCreatedEvent;
import org.springframework.stereotype.Component;

@Component
public class IssueCreatedValidator {

    public void validate(IssueCreatedEvent event) {
        if (event == null) {
            throw new IllegalArgumentException("event is null");
        }
        if (event.getIssueId() == null) {
            throw new IllegalArgumentException("issueId is missing");
        }
        if (event.getProjectId() == null) {
            throw new IllegalArgumentException("projectId is missing");
        }

        String text = event.getDescription();
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("description is missing/empty");
        }

        if (text.trim().length() < 5) {
            throw new IllegalArgumentException("description too short");
        }
    }
}
