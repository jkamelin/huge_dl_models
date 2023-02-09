import random
import argparse
import functools
import time
import torch

from icetk import icetk as tokenizer
from sr_pipeline import SRGroup 

from SwissArmyTransformer import get_args
from SwissArmyTransformer.arguments import set_random_seed
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.model import CachedAutoregressiveModel

from coglm_strategy import CoglmStrategy

tokenizer.add_special_tokens(['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

class InferenceModel(CachedAutoregressiveModel):
    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(
            logits_parallel.float(),
            self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel


def get_arguments():
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--img-size', type=int, default=160)
    py_parser.add_argument('--only-first-stage', action='store_true')
    py_parser.add_argument('--inverse-prompt', action='store_true')
    py_parser.add_argument('--style', type=str, default='mainbody', 
        choices=['none', 'mainbody', 'photo', 'flat', 'comics', 'oil', 'sketch', 'isometric', 'chinese', 'watercolor'])
    known, args_list = py_parser.parse_known_args()
    default_args = ['--mode', 'inference', '--fp16', '--output-path', 'samples_sat_v0.2']
    args_list = args_list.extend(default_args)
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known), **get_recipe(known.style))
    
    return args


def get_recipe(name):
    r = {
        'attn_plus': 1.4,
        'temp_all_gen': 1.15,
        'topk_gen': 16,
        'temp_cluster_gen': 1.,

        'temp_all_dsr': 1.5,
        'topk_dsr': 100,
        'temp_cluster_dsr': 0.89,

        'temp_all_itersr': 1.3,
        'topk_itersr': 16,
        'query_template': '{}<start_of_image>'
    }
    if name == 'none':
        pass
    elif name == 'mainbody':
        r['query_template'] = '{} 高清摄影 隔绝<start_of_image>'
        
    elif name == 'photo':
        r['query_template'] = '{} 高清摄影<start_of_image>'
        
    elif name == 'flat':
        r['query_template'] = '{} 平面风格<start_of_image>'
        # r['attn_plus'] = 1.8
        # r['temp_cluster_gen'] = 0.75
        r['temp_all_gen'] = 1.1
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4

        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'comics':
        r['query_template'] = '{} 漫画 隔绝<start_of_image>'
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4
        r['temp_all_gen'] = 1.1
        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'oil':
        r['query_template'] = '{} 油画风格<start_of_image>'
        pass
    elif name == 'sketch':
        r['query_template'] = '{} 素描风格<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'isometric':
        r['query_template'] = '{} 等距矢量图<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'chinese':
        r['query_template'] = '{} 水墨国画<start_of_image>'
        r['temp_all_gen'] = 1.12
    elif name == 'watercolor':
        r['query_template'] = '{} 水彩画风格<start_of_image>'
    return r


def get_masks_and_position_ids_coglm(seq, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[:context_length])
    torch.arange(512, 512 + len(seq) - context_length, 
            out=position_ids[context_length:]
    )

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids


def load_model(args):
    model, args = InferenceModel.from_pretrained(args, 'coglm')
    model.transformer.cpu()
    return model, args


def load_strategy(args):
    invalid_slices = [slice(tokenizer.num_image_tokens, None)]
    strategy = CoglmStrategy(invalid_slices, temperature=args.temp_all_gen,
                             top_k=args.topk_gen, top_k_cluster=args.temp_cluster_gen)

    return strategy


def load_srg(args):
    if not args.only_first_stage:
        srg = SRGroup(args)
        srg.dsr.max_bz = 2

        return srg


@torch.inference_mode()
def preprocess_text(args, text):
    text = args.query_template.format(text)

    start = time.perf_counter()
    seq = tokenizer.encode(text)

    if len(seq) > 110:
        print('The input text is too long.')
        return None, None
    txt_len = len(seq) - 1
    seq = torch.tensor(seq + [-1] * 400, device=args.device)

    elapsed = time.perf_counter() - start
    print(f'Preprocessing time: {elapsed} sec')
    return seq, txt_len


@torch.inference_mode()
def generate_tokens(args, model, strategy, seq, txt_len, num=8):
    start = time.perf_counter()

    # calibrate text length
    log_attention_weights = torch.zeros(
        len(seq),
        len(seq),
        device=args.device,
        dtype=torch.half if args.fp16 else torch.float32)
    log_attention_weights[:, :txt_len] = args.attn_plus
    get_func = functools.partial(get_masks_and_position_ids_coglm,
                                 context_length=txt_len)

    output_list = []
    remaining = num
    for _ in range((num + args.max_batch_size - 1) // args.max_batch_size):
        strategy.start_pos = txt_len + 1
        coarse_samples = filling_sequence(
            model,
            seq.clone(),
            batch_size=min(remaining, args.max_batch_size),
            strategy=strategy,
            log_attention_weights=log_attention_weights,
            get_masks_and_position_ids=get_func)[0]
        output_list.append(coarse_samples)
        remaining -= args.max_batch_size
    output_tokens = torch.cat(output_list, dim=0)

    elapsed = time.perf_counter() - start
    print(f'Token generation time: {elapsed} sec')
    return output_tokens


def postprocess(tensor):
    return tensor.cpu().mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to(torch.uint8).numpy()


@torch.inference_mode()
def generate_images(seq, txt_len, tokens, only_first_stage=True, srg=None):
    start = time.perf_counter()

    res = []
    if only_first_stage:
        for i in range(len(tokens)):
            seq = tokens[i]
            decoded_img = tokenizer.decode(image_ids=seq[-400:])
            decoded_img = torch.nn.functional.interpolate(decoded_img,
                                                          size=(480, 480))
            decoded_img = postprocess(decoded_img[0])
            res.append(decoded_img)  # only the last image (target)
    else:  # sr
        iter_tokens = srg.sr_base(tokens[:, -400:], seq[:txt_len])
        for seq in iter_tokens:
            decoded_img = tokenizer.decode(image_ids=seq[-3600:])
            decoded_img = torch.nn.functional.interpolate(decoded_img,
                                                          size=(480, 480))
            decoded_img = postprocess(decoded_img[0])
            res.append(decoded_img)  # only the last image (target)

    elapsed = time.perf_counter() - start
    print(f'Image generation time: {elapsed} sec')
    return res


def run(args, text, model, strategy, seed, only_first_stage, num, srg=None):
    start = time.perf_counter()

    set_random_seed(seed)
    seq, txt_len = preprocess_text(text)
    if seq is None:
        return None

    if not only_first_stage or srg is not None:
        srg.dsr.model.cpu()
        srg.itersr.model.cpu()
    # torch.cuda.empty_cache()
    model.transformer.to(args.device)
    tokens = generate_tokens(args, model, strategy, seq, txt_len, num)

    if not only_first_stage:
        model.transformer.cpu()
        # torch.cuda.empty_cache()
        srg.dsr.model.to(args.device)
        srg.itersr.model.to(args.device)
    # torch.cuda.empty_cache()
    res = generate_images(seq, txt_len, tokens, only_first_stage, srg)

    elapsed = time.perf_counter() - start
    print(f'Total time: {elapsed} sec')
    return res


def main():
    args = get_arguments()
    text = '0	a tiger wearing VR glasses'

    model, args = load_model(args)
    strategy = load_strategy(args)
    srg = load_srg(args)

    args.device = torch.device(args.device)

    rng = random.Random()
    seed = rng.randint(0, 100000)
    results = run(args, text, model, strategy, seed, args.only_first_stage, 8, srg)


if __name__ == '__main__':
    main()
