# © Artur Czarnecki. All rights reserved.
# LKW local Sentry proof — trimmed from getsentry/self-hosted 24.8.0 sentry.conf.example.py

from sentry.conf.server import *  # NOQA

DATABASES = {
    "default": {
        "ENGINE": "sentry.db.postgres",
        "NAME": "postgres",
        "USER": "postgres",
        "PASSWORD": "",
        "HOST": "sentry-postgres",
        "PORT": "",
    }
}

SENTRY_USE_BIG_INTS = True
SENTRY_SINGLE_ORGANIZATION = True

SENTRY_OPTIONS["system.event-retention-days"] = int(
    env("SENTRY_EVENT_RETENTION_DAYS", "7")  # noqa: F405
)

SENTRY_OPTIONS["redis.clusters"] = {
    "default": {
        "hosts": {
            0: {
                "host": "sentry-redis",
                "password": "",
                "port": "6379",
                "db": "0",
            }
        }
    }
}

BROKER_URL = "redis://sentry-redis:6379/0"

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.memcached.PyMemcacheCache",
        "LOCATION": ["sentry-memcached:11211"],
        "TIMEOUT": 3600,
        "OPTIONS": {"ignore_exc": True},
    }
}

SENTRY_CACHE = "sentry.cache.redis.RedisCache"

DEFAULT_KAFKA_OPTIONS = {
    "bootstrap.servers": "sentry-kafka:9092",
    "message.max.bytes": 50000000,
    "socket.timeout.ms": 1000,
}

SENTRY_EVENTSTREAM = "sentry.eventstream.kafka.KafkaEventStream"
SENTRY_EVENTSTREAM_OPTIONS = {"producer_configuration": DEFAULT_KAFKA_OPTIONS}
KAFKA_CLUSTERS["default"] = DEFAULT_KAFKA_OPTIONS  # noqa: F405

SENTRY_RATELIMITER = "sentry.ratelimits.redis.RedisRateLimiter"
SENTRY_BUFFER = "sentry.buffer.redis.RedisBuffer"
SENTRY_QUOTAS = "sentry.quotas.redis.RedisQuota"
SENTRY_TSDB = "sentry.tsdb.redissnuba.RedisSnubaTSDB"
SENTRY_SEARCH = "sentry.search.snuba.EventsDatasetSnubaSearchBackend"
SENTRY_SEARCH_OPTIONS = {}
SENTRY_TAGSTORE_OPTIONS = {}
SENTRY_DIGESTS = "sentry.digests.backends.redis.RedisBackend"
SENTRY_RELEASE_HEALTH = "sentry.release_health.metrics.MetricsReleaseHealthBackend"
SENTRY_RELEASE_MONITOR = (
    "sentry.release_health.release_monitor.metrics.MetricReleaseMonitorBackend"
)

SENTRY_WEB_HOST = "0.0.0.0"
SENTRY_WEB_PORT = 9000
SENTRY_WEB_OPTIONS = {
    "http": "%s:%s" % (SENTRY_WEB_HOST, SENTRY_WEB_PORT),
    "protocol": "uwsgi",
    "uwsgi-socket": None,
    "so-keepalive": True,
    "http-keepalive": 15,
    "http-chunked-input": True,
    "workers": 2,
    "threads": 2,
    "memory-report": False,
    "max-requests": 100000,
    "max-requests-delta": 500,
    "max-worker-lifetime": 86400,
    "thunder-lock": True,
    "log-x-forwarded-for": False,
    "buffer-size": 32768,
    "limit-post": 209715200,
    "disable-logging": True,
    "reload-on-rss": 600,
    "ignore-sigpipe": True,
    "ignore-write-errors": True,
    "disable-write-exception": True,
}

SENTRY_OPTIONS["mail.list-namespace"] = env("SENTRY_MAIL_HOST", "localhost")  # noqa: F405
SENTRY_OPTIONS["mail.from"] = f"sentry@{SENTRY_OPTIONS['mail.list-namespace']}"

SENTRY_FEATURES["projects:sample-events"] = False
SENTRY_SELF_HOSTED_ERRORS_ONLY = True

# GeoIP disabled for local proof stack.
GEOIP_PATH_MMDB = ""

CSRF_TRUSTED_ORIGINS = ["http://127.0.0.1:9000", "http://localhost:9000"]
