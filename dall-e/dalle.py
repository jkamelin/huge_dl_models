import argparse
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import random
from time import perf_counter

from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from functools import partial

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

DALLE_MODEL = 'dalle-mini/dalle-mini/mega-1-fp16:latest'
VQGAN_REPO = 'dalle-mini/vqgan_imagenet_f16_16384'


class DalleModel:
    def __init__(self, model, params, vqgan, vqgan_params):
        self.model = model
        self.params = params
        self.vqgan = vqgan
        self.vqgan_params = vqgan_params

    @classmethod
    def load_model(cls, dalle_model, vqgan_model, use_jax):
        model, params = DalleBart.from_pretrained(dalle_model, dtype=np.float16, _do_init=False)

        vqgan, vqgan_params = VQModel.from_pretrained(vqgan_model, _do_init=False)

        if use_jax:
            params = replicate(params)
            vqgan_params = replicate(vqgan_params)

        return cls(model, params, vqgan, vqgan_params)

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5))
    def p_generate(self, tokenized_prompt, key, top_k, top_p, condition_scale):
        return self.model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=self.params,
            top_k=top_k,
            top_p=top_p,
            condition_scale=condition_scale,
            do_sample=True
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(self, indices):
        return self.vqgan.decode_code(indices, params=self.vqgan_params)

    def generate_images_jax(self, tokenized_prompt, key, n_predictions=8,
                            gen_top_k=5, gen_top_p=0.8, cond_scale=10.0):
        images = []
        inference_time = 0
        iter_number = max(n_predictions // jax.device_count(), 1)
        for _ in range(iter_number):
            key, subkey = jax.random.split(key)

            t0 = perf_counter()
            encoded_images = self.p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                gen_top_k,
                gen_top_p,
                cond_scale,
            )
            encoded_images = encoded_images.sequences[..., 1:]
            decoded_images = self.p_decode(encoded_images)
            generation_time = perf_counter() - t0

            inference_time += generation_time
            for decoded_img in decoded_images:
                images.append(decoded_img)

        return images, inference_time/iter_number

    def generate_images(self, tokenized_prompt, condition_scale=10.0, top_k=5, top_p=0.8):
        images = []
        t0 = perf_counter()
        encoded_images = self.model.generate(**tokenized_prompt, params=self.params, top_k=top_k, top_p=top_p,
                                             condition_scale=condition_scale, do_sample=True)
        encoded_images = encoded_images.sequences[..., 1:]
        decoded_images = self.vqgan.decode_code(encoded_images, params=self.vqgan_params)
        generation_time = perf_counter() - t0

        for decoded_img in decoded_images:
            decoded_img = decoded_img.clip(0.0, 1.0) * 255
            images.append(decoded_img)

        return images, generation_time


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dalle_model', type=str, default=DALLE_MODEL,
                        help="Optional. Path to dalle model")
    parser.add_argument('--vqgan_model', type=str, default=VQGAN_REPO,
                        help="Optional. Path to vqgan model")
    parser.add_argument('--use_jax', action='store_true', help="Optional. Use jax.")
    parser.add_argument('--show', action='store_true', help="Optional. Show output.")

    args = parser.parse_args()

    return args


def prepare_input(prompts, dalle_model):
    processor = DalleBartProcessor.from_pretrained(dalle_model, revision=None)
    tokenized_prompts = processor(prompts)

    return tokenized_prompts


def main():
    args = cli_argument_parser()
    dalle = DalleModel.load_model(args.dalle_model, args.vqgan_model, args.use_jax)

    inference_time = []
    # while True:
    text_input =  [
        "sunset over a lake in the mountains",
        "the Eiffel tower landing on the moon",
    ]

    tokenized_prompts = prepare_input(text_input, args.dalle_model)
    if args.use_jax:
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
        tokenized_prompt = replicate(tokenized_prompts)
        images, time_ = dalle.generate_images_jax(tokenized_prompt, key)
    else:
        images, time_ = dalle.generate_images(tokenized_prompts)
    inference_time.append(time_)

    if args.show:
        for img in images:
            img = np.asarray(img, dtype=np.uint8)
            cv2.imshow(img)

    avg_time = np.average(inference_time)
    print(f'Average inference time: {avg_time} seconds')


if __name__ == '__main__':
    main()
