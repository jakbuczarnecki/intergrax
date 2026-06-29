# Kafka (kafka)

Category: `message_bus`

## Single public entrypoint

- **`KafkaMessageBusIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `KafkaMessageBusIntegration`.
- Contract factory: `create_kafka_message_bus_integration()`.
