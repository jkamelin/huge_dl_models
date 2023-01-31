import argparse
import cv2
import numpy as np
from time import perf_counter

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

DALLE_MODEL = '/home/jkamelin/yadro/huge_dl_models/dall-e/dalle-mini'
DALLE_COMMIT_ID = None

VQGAN_REPO = '/home/jkamelin/yadro/huge_dl_models/dall-e/vqgan_imagenet_f16_16384'
VQGAN_COMMIT_ID = None


class DalleModel:
    def __init__(self, model, params, vqgan, vqgan_params):
        self.model = model
        self.params = params
        self.vqgan = vqgan
        self.vqgan_params = vqgan_params

    @classmethod
    def load_model(cls):
        model, params = DalleBart.from_pretrained(DALLE_MODEL, dtype=np.float16, _do_init=False)

        vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, _do_init=False)

        return cls(model, params, vqgan, vqgan_params)

    def generate_images(self, tokenized_prompt, condition_scale, top_k, top_p):
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

    parser.add_argument('--show', action='store_true', help="Optional. Show output.")
    parser.add_argument('--batch_size', type=int, default=9, help="Batch size")

    args = parser.parse_args()

    return args


def prepare_input(prompts):
    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
    tokenized_prompts = processor(prompts)

    return tokenized_prompts


def main():
    args = cli_argument_parser()

    dalle = DalleModel.load_model()

    prompts = [
        "sunset over a lake in the mountains",
        "the Eiffel tower landing on the moon",
        "sunflower on a snowboard",
        "storm in a glass",
        "hamster on a treadmill",
        "winter in a desert",
        "Sasha walking along the highway",
        "pank on a wedding",
        "skateboarder's keyboard"
    ]

    inference_time = []
    for i in range(0, len(prompts), args.batch_size):
        tokenized_prompts = prepare_input(prompts[i:i+args.batch_size])
        images, time_ = dalle.generate_images(tokenized_prompts, 10.0, 5, 0.8)
        inference_time.append(time_)

        if args.show:
            for img in images:
                img = np.asarray(img, dtype=np.uint8)
                cv2.imshow(img)

    avg_time = np.average(inference_time)
    print(f'Average inference time: {avg_time} seconds')


if __name__ == '__main__':
    main()
