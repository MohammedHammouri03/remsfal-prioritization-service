package de.remsfal.prioritization.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class HttpClientConfig {

    @Bean
    public WebClient inferenceWebClient(InferenceConfig cfg) {
        return WebClient.builder()
                .baseUrl(cfg.getBaseUrl())
                .build();
    }
}
