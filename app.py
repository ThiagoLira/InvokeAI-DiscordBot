import asyncio
import sys
from log import get_logger
from dreambot import DreamBot

logger = get_logger(__name__)


async def shutdown(bot):
    await bot.close()

def main():
    dreambot = None

    with open('.token', 'r') as f:
        TOKEN = f.read()

    try:
        dreambot = DreamBot()
        dreambot.run(TOKEN)
    except KeyboardInterrupt:
        logger.info('Keyboard interrupt received. Exiting.')
        asyncio.run(shutdown(dreambot))
    except SystemExit:
        logger.info('System exit received. Exiting.')
        asyncio.run(shutdown(dreambot))
    except Exception as e:
        logger.error(e)
        asyncio.run(shutdown(dreambot))
    finally:
        sys.exit(0)


if __name__ == '__main__':
    main()
