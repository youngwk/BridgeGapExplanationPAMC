from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math 

   

'''
loss functions
'''

def loss_an(logits, observed_labels):

    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(), reduction='none')
    return loss_matrix, corrected_loss_matrix


'''
top-level wrapper
'''

def compute_batch_loss(logits, label_vec, P): 
     
    assert logits.dim() == 2
    
    batch_size = int(logits.size(0))
    num_classes = int(logits.size(1))
    

    if P['dataset'] == 'OPENIMAGES':
        unobserved_mask = (label_vec == -1)
    else:
        unobserved_mask = (label_vec == 0)
    
    # compute loss for each image and class:
    loss_matrix, corrected_loss_matrix = loss_an(logits, label_vec.clip(0))

    correction_idx = [torch.Tensor([]), torch.Tensor([])]

    if P['clean_rate'] == 1: # if epoch is 1, do not modify losses
        final_loss_matrix = loss_matrix
    else:
        if P['largelossmod_scheme'] == 'LL-Cp':
            k = math.ceil(batch_size * num_classes * P['delta_rel'])
        else:
            k = math.ceil(batch_size * num_classes * (1-P['clean_rate']))
    
        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]
        correction_idx = torch.where(unobserved_loss >= topk_lossvalue)


        if P['largelossmod_scheme'] in ['LL-Ct', 'LL-Cp']:
            final_loss_matrix = torch.where(unobserved_loss >= topk_lossvalue, corrected_loss_matrix, loss_matrix)
        else:
            zero_loss_matrix = torch.zeros_like(loss_matrix)
            final_loss_matrix = torch.where(unobserved_loss >= topk_lossvalue, zero_loss_matrix, loss_matrix)
                
    main_loss = final_loss_matrix.mean()
    
    return main_loss, correction_idx