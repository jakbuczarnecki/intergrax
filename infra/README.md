# Intergrax Infrastructure Layer

This directory contains infrastructure modules used by the Intergrax platform.

Each component (e.g., Redis, Kafka, RabbitMQ) is self-contained and provides:

- docker-compose.yml
- manage.ps1 (Windows lifecycle script)
- manage.sh (Linux / CI lifecycle script)

Infrastructure is fully decoupled from runtime code (`intergrax/`).

---

## Redis

Location:

infra/docker/redis/

### Start (Windows)

```powershell
.\manage.ps1 start