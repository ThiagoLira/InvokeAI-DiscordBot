import traceback
import string
from torch import autocast
import requests
import discord
from discord.ext import commands, tasks
from typing import Optional
from io import BytesIO
from PIL import Image
from discord import Message, option
import torch
from typing import List, Dict, Tuple
import os
import aiohttp



embed_color = discord.Colour.from_rgb(215, 195, 134)

PROMPT_QUEUE : List[Tuple[Dict,discord.Message]] = []
SERVER_URL = 'http://192.168.15.176:5000'



class StableCog(commands.Cog, name='Stable Diffusion', description='Create images from natural language.'):
    def __init__(self, bot):
        self.bot = bot
        self.generator_loop.start()

    @tasks.loop(seconds=5.0)
    async def generator_loop(self):
        global PROMPT_QUEUE

        if len(PROMPT_QUEUE) > 0:
            async with aiohttp.ClientSession() as session:
                arg_dict, channel = PROMPT_QUEUE.pop(0)
                
                embed = discord.Embed()
                embed.color = embed_color
                embed.set_footer(text=arg_dict['prompt'])
                arg_dict['outdir'] = './outputs/' 

                if 'init_image' in arg_dict:
                    import random 

                    img_path = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png'
                    arg_dict['init_image'].save(img_path) 
                    arg_dict['init_image'] = img_path

                async with session.get(SERVER_URL, params=arg_dict, timeout=60) as resp:
                    if resp.status==400:
                        PROMPT_QUEUE.append((arg_dict,channel))
                        await channel.send('DEU ERRO')
                        return

                    bytes_ = await resp.read()

                    await channel.send(embed=embed, file=discord.File(fp=BytesIO(bytes_), filename=f'{arg_dict["prompt"]}.png'))

                    if 'init_image' in arg_dict:
                        try:
                            os.remove(img_path)
                        except:
                            print('Vc fez bosta')


    @discord.slash_command(name='dream')
    @option(
        'prompt',
        str,
        description = 'A prompt to condition the model with.',
        required=True,
    )
    @option(
        'height',
        int,
        description = 'Height of the generated image.',
        required = False,
        choices = [x for x in range(192, 832, 64)]
    )
    @option(
        'width',
        int,
        description = 'Width of the generated image.',
        required = False,
        choices = [x for x in range(192, 832, 64)]
    )
    @option(
        'guidance_scale',
        float,
        description = 'Classifier-Free Guidance scale',
        required = False,
    )
    @option(
        'steps',
        int,
        description = 'The amount of steps to sample the model',
        required = False,
        choices = [x for x in range(5, 105, 5)]
    )
    @option(
        'seed',
        int,
        description = 'The seed to use for reproduceability',
        required = False,
    )
    @option(
        'strength',
        float,
        description = 'The strength used to apply the prompt to the init_image/mask_image'
    )
    @option(
        'init_image',
        discord.Attachment,
        description = 'The image to initialize the latents with for denoising',
        required = False,
    )
    @option(
        'mask_image',
        discord.Attachment,
        description = 'The mask image to use for inpainting',
        required = False,
    )
    async def dream(self, ctx: discord.ApplicationContext, *, query: str, height: Optional[int]=512, width: Optional[int]=512, guidance_scale: Optional[float] = 7.0, steps: Optional[int] = 50, seed: Optional[int] = -1, strength: Optional[float]=0.8, init_image: Optional[discord.Attachment] = None, mask_image: Optional[discord.Attachment] = None):
        print(f'Request -- {ctx.author.name}#{ctx.author.discriminator} -- Prompt: {query}')
        global PROMPT_QUEUE
        await ctx.defer()
        try:
            '''
                iterations     = iterations,
                seed           = self.seed,
                sampler        = self.sampler,
                steps          = steps,
                cfg_scale      = cfg_scale,
                conditioning   = (uc,c),
                ddim_eta       = ddim_eta,
                image_callback = image_callback,  # called after the final image is generated
                step_callback  = step_callback,   # called after each intermediate image is generated
                width          = width,
                height         = height,
                init_image     = init_image,      # notice that init_image is different from init_img
                mask_image     = mask_image,
                strength       = strength,
            '''
            if (init_image is None) and (mask_image is None):
                arg_dict = {
                    'prompt' : query,
                    'steps' : steps,
                    'cfg_scale' : guidance_scale,
                    'strength' : strength,
                    'seed' : seed,
                    'height' : height,
                    'width' : width,
                }
                PROMPT_QUEUE.append((arg_dict, ctx.channel))
                embed = discord.Embed(title='Prompt adicionado na fila!',  color=embed_color)
                await ctx.send_followup(embed=embed, ephemeral=True)
            elif (init_image is not None):
                image = Image.open(requests.get(init_image.url, stream=True).raw).convert('RGB')
                arg_dict = {
                    'prompt' : query,
                    'init_image' : image,
                    'steps' : steps,
                    'cfg_scale' : guidance_scale,
                    'strength' : strength,
                    'height' : height,
                    'seed' : seed,
                    'width' : width,
                }
                PROMPT_QUEUE.append((arg_dict, ctx.channel))
                embed = discord.Embed(title='Prompt adicionado na fila!',  color=embed_color )
                await ctx.send_followup(embed=embed, ephemeral=True)
            else:
                raise Exception('Ainda nao implementado')
                image = Image.open(requests.get(init_image.url, stream=True).raw).convert('RGB')
                mask = Image.open(requests.get(mask_image.url, stream=True).raw).convert('RGB')
                samples, seed = self.image2image_model.inpaint(query, image, mask, steps, 0.0, 1, 1, guidance_scale, denoising_strength=strength, seed=seed, height=height, width=width)

        except Exception as e:
            embed = discord.Embed(title='txt2img failed', description=f'{e}\n{traceback.print_exc()}', color=embed_color)
            await ctx.followup.send(embed=embed)


def setup(bot):
    bot.add_cog(StableCog(bot))
