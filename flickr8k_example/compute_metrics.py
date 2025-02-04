'''
Computes the metrics for Flickr8K.
'''
import sys
sys.path.append('../')
import clipscore
import generation_eval_utils
import scipy.stats
import os
import json
import numpy as np
import torch
import warnings
import clip


def compute_human_correlation(input_json, image_directory, tauvariant='c'):

    data = {}
    with open(input_json) as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    images = []
    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue
            images.append(image_directory + '/' + v['image_path'])
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))
            human_scores.append(human_judgement['rating'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = clipscore.extract_all_images(
        images, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore (dot product)
    _dot, per_instance_image_text_dot, candidate_feats_dot = clipscore.get_clip_score(
        model, image_feats, candidates, device)
    
    # get image-text clipscore (cosine )
    _cos, per_instance_image_text_cos, candidate_feats_cos = clipscore.get_clip_score_cosine(
        model, image_feats, candidates, device)

    # F-score
    # refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
    # other_metrics = generation_eval_utils.get_all_metrics(refs, candidates, return_per_cap=True)

    print('CLIPScore dot Tau-{}: {:.3f}'.format(tauvariant, 100*scipy.stats.kendalltau(per_instance_image_text_dot, human_scores, variant=tauvariant)[0]))
    print('CLIPScore cos Tau-{}: {:.3f}'.format(tauvariant, 100*scipy.stats.kendalltau(per_instance_image_text_cos, human_scores, variant=tauvariant)[0]))
   

def main():
    if not os.path.exists('flickr8k/flickr8k.json'):
        print('Please run download.py')
        quit()
    print('Flickr8K Expert (Tau-c)')
    compute_human_correlation('flickr8k/flickr8k.json', 'flickr8k/', tauvariant='c')

    print('Flickr8K CrowdFlower (Tau-b)')
    compute_human_correlation('flickr8k/crowdflower_flickr8k.json', 'flickr8k/', tauvariant='b')


if __name__ == '__main__':
    main()
