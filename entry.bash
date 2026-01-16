#!/bin/bash
python manage.py collectstatic --noinput

# start gunicorn (bind to all interfaces on 8000)
exec gunicorn backend.wsgi:application --bind 0.0.0.0:8000