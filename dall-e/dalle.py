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

DALLE_MODEL = 'dalle-mini/dalle-mini/mini-1:v0'
VQGAN_REPO = 'dalle-mini/vqgan_imagenet_f16_16384'


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dalle_model', type=str, default=DALLE_MODEL,
                        help="Optional. Path to dalle model")
    parser.add_argument('--vqgan_model', type=str, default=VQGAN_REPO,
                        help="Optional. Path to vqgan model")
    parser.add_argument('--n_predictions', type=int, default=8,
                        help="Optional. Number of predictions to generate")
    parser.add_argument('--top_k', type=int, default=5,
                        help="Optional. Number of predictions with the highest probability")
    parser.add_argument('--top_p', type=float, default=0.8,
                        help="Optional. Maximum probability for predictions")
    parser.add_argument('--cond_scale', type=float, default=10.0,
                        help="Optional. Number of predictions to generate")
    parser.add_argument('--show', action='store_true', help="Optional. Show output.")

    args = parser.parse_args()

    return args


def load_model(dalle_model, vqgan_model):
    model, params = DalleBart.from_pretrained(dalle_model, dtype=np.float16, _do_init=False)
    vqgan, vqgan_params = VQModel.from_pretrained(vqgan_model, _do_init=False)

    params = replicate(params)
    vqgan_params = replicate(vqgan_params)

    return model, params, vqgan, vqgan_params


def generate_images(model, params, vqgan, vqgan_params, tokenized_prompt, n_predictions,
                    cond_scale, gen_top_k, gen_top_p):
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5))
    def p_generate(tokenized_prompt, key, params, top_k, top_p, condition_scale):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            condition_scale=condition_scale,
        )

    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, vqgan_params):
        return vqgan.decode_code(indices, params=vqgan_params)

    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)

    images = []
    inference_time = 0
    iter_number = max(n_predictions // jax.device_count(), 1)
    for _ in range(iter_number):
        key, subkey = jax.random.split(key)

        t0 = perf_counter()
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            cond_scale,
        )
        encoded_images = encoded_images.sequences[..., 1:]
        decoded_images = p_decode(encoded_images, vqgan_params)
        generation_time = perf_counter() - t0

        inference_time += generation_time
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for decoded_img in decoded_images:
            images.append(decoded_img * 255)

    return images, inference_time/iter_number


def prepare_input(prompt, dalle_model):
    processor = DalleBartProcessor.from_pretrained(dalle_model, revision=None)
    tokenized_prompts = processor(prompt)
    tokenized_prompt = replicate(tokenized_prompts)

    return tokenized_prompt


def main():
    args = cli_argument_parser()
    model, params, vqgan, vqgan_params = load_model(args.dalle_model, args.vqgan_model)

    prompts = [
        "sunset over a lake in the mountains",
        "the Eiffel tower landing on the moon",
    ]

    tokenized_prompt = prepare_input(prompts, args.dalle_model)
    images, time = generate_images(model, params, vqgan, vqgan_params, tokenized_prompt,
                                   args.n_predictions, args.cond_scale, args.top_k, args.top_p)

    print(f'Average inference time: {time} seconds')
    if args.show:
        for img in images:
            img = np.asarray(img, dtype=np.uint8)
            cv2.imshow(img)


if __name__ == '__main__':
    main()
