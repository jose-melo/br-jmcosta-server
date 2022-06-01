import logging
import config
import os

from .modules.puissance4.GetMove import get_move

logging.basicConfig(level=config.LOGLEVEL)
logger = logging.getLogger(__name__)


def process_connect_four(payload: dict) -> dict:
    logging.info('payload: '+str(payload))

    move = get_move(payload['actions']) 

    logging.info('move: '+str(move))
    return {"move": move}