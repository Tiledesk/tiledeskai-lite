#!/bin/bash
set -euo pipefail

# Configurazione predefinita
WORKERS=3
TIMEOUT=240
MAXREQUESTS=2000
MAXRJITTER=5
GRACEFULTIMEOUT=60

# Analisi degli argomenti della riga di comando (solo --help)
myargs=$(getopt --name "$0" -o h --long "help" -- "$@")
eval set -- "$myargs"

while true; do
    case "$1" in
        -h|--help)
            echo "usage $0 --help"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        --)
            shift
            break
            ;;
    esac
done

# Verifica dell'installazione di Gunicorn
if ! command -v gunicorn &> /dev/null; then
    echo "Error: Gunicorn is not installed."
    exit 1
fi

# Verifica esistenza file di log
if [ ! -f "log_conf.json" ]; then
  echo "Error: log_conf.json file not found"
  exit 1
fi

# Avvio di Gunicorn
echo "Starting Gunicorn with workers: $WORKERS, timeout: $TIMEOUT, max requests: $MAXREQUESTS, max jitter: $MAXRJITTER, graceful timeout: $GRACEFULTIMEOUT"

gunicorn --bind 0.0.0.0:8000 --workers "$WORKERS" --timeout "$TIMEOUT" --max-requests "$MAXREQUESTS" --max-requests-jitter "$MAXRJITTER" --graceful-timeout "$GRACEFULTIMEOUT" --log-config-json log_conf.json --worker-class uvicorn.workers.UvicornWorker tilelite.__main__:app