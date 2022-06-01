import logging
import os

from app import flask


# pr = cProfile.Profile()
# pr.enable()
APP = flask.create_app()

# def f():
#     pr.disable()
#     pr.dump_stats("stats.prof")
#     pr.print_stats()

# atexit.register(f)

if __name__ == "__main__":
    print('lkashfdjlsdkjhfalskj')
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting API...")
    try:
        APP.run(
            host="0.0.0.0",
            port=os.environ.get("PORT", 8080),
            debug=True,
            use_reloader=True,
        )
    except Exception as err:
        import sys
        print(err, file=sys.stderr)
