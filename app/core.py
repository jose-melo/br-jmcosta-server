import logging
import config

from .modules.puissance4.GetMove import get_move

logging.basicConfig(level=config.LOGLEVEL)
logger = logging.getLogger(__name__)


def process_connect_four(payload: dict) -> dict:
    print('payload: ', payload)

    move = get_move(payload['actions']) 

    print('move: ', move)

    return {"move": move}