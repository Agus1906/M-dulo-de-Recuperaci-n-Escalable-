import logging
import sys


def run(action):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # SDK exceptions can include response details: expose only the exception type at the boundary.
    try:
        action()
    except (ValueError, FileNotFoundError) as exc:
        logging.error("Validación: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.error(
            "Falló la operación (%s). Revisar credenciales, red y estado del servicio; "
            "es seguro reintentar la ingesta.",
            type(exc).__name__,
        )
        sys.exit(1)
