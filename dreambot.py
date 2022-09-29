import discord
from discord.ext import commands
from log import get_logger

class DreamBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.members = True
        super().__init__(command_prefix=commands.when_mentioned_or('?'), intents=intents, auto_sync_commands=True)
        self.logger = get_logger(__name__)
        self.load_extension('stablecog')

    async def on_ready(self):
        self.logger.info(f'Logged in as {self.user.name} ({self.user.id})')
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='Vendo as coisas degen que vocês geram...'))
        print('loaded')
