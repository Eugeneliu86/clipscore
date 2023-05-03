import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import clipscore
import clip
import torch
import warnings



def filter_data(input_json, image_directory):
    # unpacking JSON file
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
        
    # enable GPU    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    
    
    # running clip score
    image_feats = clipscore.extract_all_images(
        images, model, device, batch_size=64, num_workers=8)
    
    dot, per_instance_image_text_dot, candidate_feats_dot = clipscore.get_clip_score(
        model, image_feats, candidates, device)

    
    # plotting pie chart + percentage of likert score distribution

    likert_score_counts = np.zeros(len(set(human_scores)))
    print(human_scores)
    
    for value in human_scores:
        likert_score_counts[int(value-1)] +=1
    
    likert_score_percentages = likert_score_counts/np.sum(likert_score_counts) * 100
    
    plt.pie(likert_score_percentages, labels=list(range(len(set(human_scores)))),
             autopct='%1.2f%%')

    plt.title("Distribution of Likert Scores")
    plt.savefig("likert_score_distribution")
    
    # sorting scores and partitioning
    ranked_scores = np.sort(per_instance_image_text_dot)
    
    print(ranked_scores)
    score_cutt_offs = []
    for i in range(len(likert_score_counts)):
        score_cutt_offs.append(ranked_scores[int(sum(likert_score_counts[:i+1])-1)])
    
    print(score_cutt_offs)
    percentage = []
    
    for i in range(len(likert_score_counts)):
        if i == 0:
            prev_val = 0
        else:
            prev_val = likert_score_counts[i-1]
        current_val = likert_score_counts[i]
        
        max_val = score_cutt_offs[i]
        
        print(len(ranked_scores[int(prev_val): int(current_val)]))
        
        counter = 0
        for score in ranked_scores[int(prev_val): int(current_val)]:
            if score <= max_val:
                counter += 1
        percentage.append(counter)
    
    percentage = np.array(percentage)
    print(percentage)
    print(percentage / np.array(likert_score_counts))
        

def main():
    filter_data('flickr8k/flickr8k.json', 'flickr8k/')
    
if __name__ == '__main__':
    main()
            
            
            
            
            
            
    
    
    