package de.remsfal.prioritization.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "remsfal.kafka.topics")
public class KafkaTopicsConfig {

    private String in;
    private String out;

    public String getIn() { return in; }
    public void setIn(String in) { this.in = in; }

    public String getOut() { return out; }
    public void setOut(String out) { this.out = out; }
}
