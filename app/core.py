import logging
import config
import os

#from .modules.puissance4.GetMove import get_move

logging.basicConfig(level=config.LOGLEVEL)
logger = logging.getLogger(__name__)


def process_connect_four(payload: dict) -> dict:
    print('payload: ', payload)

    #move = get_move(payload['actions']) 

    #print('move: ', move)
    print('test: ', os.getcwd())
    print('dir: ', os.listdir())
    os.chdir('/back_end/assets/data/daily_donkey/save48.h5')
    print('test: ', os.getcwd())
    print('dir: ', os.listdir())
    return {"move": 0}