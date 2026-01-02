package de.remsfal.prioritization;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import de.remsfal.prioritization.config.KafkaTopicsConfig;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@EnableConfigurationProperties(KafkaTopicsConfig.class)
@SpringBootApplication
public class PrioritizationServiceApplication {

	public static void main(String[] args) {
		SpringApplication.run(PrioritizationServiceApplication.class, args);
	}

}
