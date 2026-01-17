#!/bin/bash
python manage.py collectstatic --noinput

# start gunicorn (bind to all interfaces). Default to 8000 if PORT not set
PORT=${PORT:-8000}
TIMEOUT=${TIMEOUT:-180}
exec gunicorn backend.wsgi:application --bind 0.0.0.0:${PORT} --timeout ${TIMEOUT}