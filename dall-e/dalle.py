import argparse
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import random

# from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
# from functools import partial
from time import time

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

DALLE_MODEL = 'dalle-mini/dalle-mini/mini-1:v0'
DALLE_COMMIT_ID = None

VQGAN_REPO = 'dalle-mini/vqgan_imagenet_f16_16384'
VQGAN_COMMIT_ID = 'e93a26e7707683d349bf5d5c41c5b0ef69b677a9'

# NUM_DEVICES = jax.local_device_count()


class DalleModel:
    def __init__(self, model, params, vqgan, vqgan_params):
        self.model = model
        self.params = params
        self.vqgan = vqgan
        self.vqgan_params = vqgan_params

    @classmethod
    def load_model(cls):
        model, params = DalleBart.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False)

        vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False)

        # params = replicate(params)
        # vqgan_params = replicate(vqgan_params)

        return cls(model, params, vqgan, vqgan_params)

    # model inference
    def p_generate(self, tokenized_prompt, key, top_k, top_p, temperature, condition_scale):
        return self.model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=self.params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    def p_decode(self, indices):
        return self.vqgan.decode_code(indices, params=self.vqgan_params)

    def generate_images(self, tokenized_prompt, key, n_predictions=8,
                        gen_top_k=None, gen_top_p=None, temperature=None, cond_scale=1.0):
        images = []
        inference_time = 0
        for _ in range(max(n_predictions, 1)):
            key, subkey = jax.random.split(key)

            t0 = time()
            encoded_images = self.model.generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            encoded_images = encoded_images.sequences[..., 1:]
            decoded_images = self.vqgan.decode_code(encoded_images, params=self.vqgan_params)
            generation_time = time() - t0

            inference_time += generation_time
            for decoded_img in decoded_images:
                images.append(decoded_img)

        return images, inference_time


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--show', action='store_true', help="Optional. Show output.")

    args = parser.parse_args()

    return args


def prepare_input(prompts):
    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
    tokenized_prompts = processor(prompts)
    # tokenized_prompt = replicate(tokenized_prompts)

    return tokenized_prompts


def main():
    args = cli_argument_parser()

    dalle = DalleModel.load_model()

    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    prompts = [
        "sunset over a lake in the mountains",
        "the Eiffel tower landing on the moon",
    ]

    tokenized_prompt = prepare_input(prompts)
    images, inference_time = dalle.generate_images(tokenized_prompt, key)

    avg_time = np.average(inference_time)
    print(f'Average inference time: {avg_time} seconds')

    if args.show:
        images = images.clip(0.0, 1.0).reshape((-1, 256, 256, 3)) * 255
        for img in images:
            img = np.asarray(img * 255, dtype=np.uint8)
            cv2.imshow(img)


if __name__ == '__main__':
    main()
