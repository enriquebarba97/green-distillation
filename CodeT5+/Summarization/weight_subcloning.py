import os
import math
import time
import random
import tqdm
import torch
import logging
import warnings
import numpy as np
import torch.nn.functional as F

from utils import OnlineDistilledDataset

def collect_activations(model, dataset, tokenizer, batch_size=16, max_tokens=80000, input_length=320, output_length=128):
    hooks = []
    activations_encoder = {
        'resid': {},
        'attn': {},
        'out': {},
        'in': {},
        'main': {},
        # 'interim': {}
    }

    activations_decoder = {
        'resid': {},
        'attn': {},
        'cross_attn': {},
        'out': {},
        'in': {},
        'main': {},
        # 'interim': {}
    }

    activations_shared = {
        'main': {}
    }

    def get_activation(name, layer_type, isinput, is_decoder=False):
        if name == 'embed' or name == 'unembed':
            def hook(model, input, output):
                if name not in activations_shared[layer_type]:
                    activations_shared[layer_type][name] = []
                if isinput == "input":
                    activations_shared[layer_type][name].append(input[0].sum(dim=0).detach().cpu())
                elif isinput == "output2":
                    activations_shared[layer_type][name].append(output[3].var(dim=-1).sum(dim=0).detach().cpu())
                elif isinput == "output":
                    if output.shape[1] == input_length:
                        activations_shared[layer_type][name].append(output.sum(dim=0).detach().cpu())
            return hook
        if is_decoder:
            def hook(model, input, output):
                if name not in activations_decoder[layer_type]:
                    activations_decoder[layer_type][name] = []
                if isinput == "input":
                    activations_decoder[layer_type][name].append(input[0].sum(dim=0).detach().cpu())
                elif isinput == "output2":
                    activations_decoder[layer_type][name].append(output[3].var(dim=-1).sum(dim=0).detach().cpu())
                elif isinput == "output":
                    activations_decoder[layer_type][name].append(output.sum(dim=0).detach().cpu())
        else:
            def hook(model, input, output):
                if name not in activations_encoder[layer_type]:
                    activations_encoder[layer_type][name] = []
                if isinput == "input":
                    activations_encoder[layer_type][name].append(input[0].sum(dim=0).detach().cpu())
                elif isinput == "output2":
                    activations_encoder[layer_type][name].append(output[3].var(dim=-1).sum(dim=0).detach().cpu())
                elif isinput == "output":
                    activations_encoder[layer_type][name].append(output.sum(dim=0).detach().cpu())
        return hook

    hooks.append(model.encoder.final_layer_norm.register_forward_hook(get_activation('norm', 'main', 'output')))
    hooks.append(model.decoder.final_layer_norm.register_forward_hook(get_activation('norm', 'main', 'output', is_decoder=True)))
    hooks.append(model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.register_forward_hook(get_activation('bias', 'main', 'output')))
    hooks.append(model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias.register_forward_hook(get_activation('bias', 'main', 'output', is_decoder=True)))

    hooks.append(model.shared.register_forward_hook(get_activation('embed', 'main', 'output')))
    hooks.append(model.lm_head.register_forward_hook(get_activation('unembed', 'main', 'input')))

    for i, layer in enumerate(model.encoder.block):
        layer = layer.layer
        registerable_hooks = [
            (layer[0].SelfAttention.o, "resid" , 'oprj', "output"),
            (layer[0].layer_norm, "resid", 'inorm', "output"),
            (layer[1].DenseReluDense.wo, "resid", 'dprj', "output"),
            #(layer[1].DenseReluDense.wi, "resid", 'gprj', "output"),
            (layer[0].SelfAttention, "attn", '', "output2"),

        ]
        def add_if_not_taken(module, type, name, layer_type):
            # check by name + layer_type
            if module == layer[0].SelfAttention:
                return
            registerable_hooks.append((module, type, name, layer_type))
        
        
        for name, module in layer.named_modules():
            if ("SelfAttention" in name):
                continue
            if len(name) == 0:
                continue
            if isinstance(module, torch.nn.Linear):
                add_if_not_taken(module, "out", name, "output")
                add_if_not_taken(module, "in", name, "input")
            if "norm" in name:
                add_if_not_taken(module, "out", name, "output")

        for module, type, name, layer_type in registerable_hooks:
            hooks.append(module.register_forward_hook(get_activation(f'{i}_{name}', type, layer_type)))

    for i, layer in enumerate(model.decoder.block):
        layer = layer.layer
        registerable_hooks = [
            (layer[0].SelfAttention.o, "resid" , 'oprj', "output"),
            (layer[1].EncDecAttention.o, "resid" , 'oprj', "output"),
            (layer[0].layer_norm, "resid", 'inorm', "output"),
            (layer[2].DenseReluDense.wo, "resid", 'dprj', "output"),
            #(layer[2].DenseReluDense.wi, "resid", 'gprj', "output"),
            (layer[0].SelfAttention, "attn", '', "output2"),
            (layer[1].EncDecAttention, "cross_attn" , '', "output2")

        ]
        def add_if_not_taken(module, type, name, layer_type):
            # check by name + layer_type
            if module == layer[0].SelfAttention or module == layer[1].EncDecAttention:
                return
            registerable_hooks.append((module, type, name, layer_type))
        
        
        for name, module in layer.named_modules():
            if ("SelfAttention" in name) or ("EncDecAttention" in name):
                continue
            if len(name) == 0:
                continue
            if isinstance(module, torch.nn.Linear):
                add_if_not_taken(module, "out", name, "output")
                add_if_not_taken(module, "in", name, "input")
            if "norm" in name:
                add_if_not_taken(module, "out", name, "output")

        for module, type, name, layer_type in registerable_hooks:
            hooks.append(module.register_forward_hook(get_activation(f'{i}_{name}', type, layer_type, is_decoder=True)))

    sequence_lengths = [input_length]  # Sequence lengths: 1, 2, 4, 8, ..., 2048
    total_tokens = 0
    batch_texts = []
    batch_labels = []
    batch_lengths = []
    
    # Initialize progress bar
    with tqdm.tqdm(total=max_tokens, desc='Collecting activations', unit='tokens') as progress:
        for i,sample in enumerate(dataset):
            # print("starting with sample: ", i+1, " total tokens: ", total_tokens)
            if total_tokens >= max_tokens:
                print("Maximal tokens reached, total tokens: ", total_tokens)
                break
            # report which sample item we're in using tqdm report next to the progress bar
            progress.set_postfix({
                'sample': f"{i+1}/211m",
            })

            text = sample[3]
            target = sample[4]

            for seq_len in sequence_lengths:
                if not (len(tokenizer.encode(text)) >= seq_len):
                    continue
                if random.random() > 0.9:
                    continue

         
                batch_texts.append(text)
                batch_labels.append(target)
                batch_lengths.append(seq_len)

                # Process in batches
                if len(batch_texts) >= batch_size:
                    
                    inputs = tokenizer(batch_texts, return_tensors='pt', padding="max_length", truncation=True, max_length=320)
                    labels = tokenizer(batch_labels, return_tensors='pt', padding="max_length", truncation=True, max_length=128)
                    total_tokens += inputs['input_ids'].numel()
                    
                    progress.update(inputs['input_ids'].numel())
                    with torch.no_grad():
                        out = model(**inputs, output_attentions=True, labels=labels['input_ids'])

                    batch_texts = []
                    batch_labels = []
                    batch_lengths = []
                    del inputs
                    del out                    

                if total_tokens >= max_tokens:
                    print("Maximal tokens reached, total tokens: ", total_tokens)
                    break

        # Process any remaining texts in the batch
        print("SOME TEXTS LEFT, NUMBER OF TEXTS LEFT: ", len(batch_texts))
        
    print("done with collecting activations, now concatenating them")
    # Convert activations to numpy arrays
    for layer_type in activations_encoder:
        for layer_index in activations_encoder[layer_type]:
            activations_encoder[layer_type][layer_index] = torch.stack(activations_encoder[layer_type][layer_index], dim=0)
    
    for layer_type in activations_decoder:
        for layer_index in activations_decoder[layer_type]:
            activations_decoder[layer_type][layer_index] = torch.stack(activations_decoder[layer_type][layer_index], dim=0)
    
    for layer_type in activations_shared:
        for layer_index in activations_shared[layer_type]:
            activations_shared[layer_type][layer_index] = torch.stack(activations_shared[layer_type][layer_index], dim=0)

    for hook in hooks:
        hook.remove()

    return activations_encoder, activations_decoder, activations_shared


def compute_global_neuron_importance(activations):
    neuron_importancee = []
    global_neuron_importance = torch.zeros(list(activations['resid'].values())[0].shape[-1])
    for layer_key in activations['resid']:
        layer_activations = activations['resid'][layer_key]
        neuron_importancee.append(layer_activations.abs().mean(dim=(0, 1)))
        global_neuron_importance += layer_activations.abs().mean(dim=(0, 1))
    return global_neuron_importance


def compute_head_importance(activations, num_heads):
    head_importance = []
    for layer_key in activations['attn']: 
        acts = activations['attn'][layer_key] # (B, H, S)
        acts = acts.sum(dim=-1).mean(dim=0) # (B, H, S)

        head_importance.append(acts)


    return torch.stack(head_importance, dim=0)

def prepare_attention_weights(old_attn):
    # Repeat K and V weights to match Q
    d_kv = old_attn.key_value_proj_dim
    num_heads = old_attn.n_heads
    q_weights = old_attn.q.weight.data.view(num_heads, d_kv, -1)
    k_weights = old_attn.k.weight.data.view(num_heads, old_attn.key_value_proj_dim, -1)
    v_weights = old_attn.v.weight.data.view(num_heads, old_attn.key_value_proj_dim, -1)
    o_weights = old_attn.o.weight.data.view(-1, num_heads, d_kv)

    
    return q_weights, k_weights, v_weights, o_weights




def subclone_attention(old_attn, new_attn, neuron_indices, head_indices):   
    # print (new_attn.q_proj.weight.data.shape) 
    num_kv_heads = old_attn.key_value_proj_dim
    new_num_kv_heads = new_attn.key_value_proj_dim
    repeat_factor = old_attn.n_heads // num_kv_heads

    q_weights, k_weights, v_weights, o_weights = prepare_attention_weights(old_attn)

    q_weights = q_weights[head_indices][:, :new_num_kv_heads, neuron_indices]
    k_weights = k_weights[head_indices][:, :new_num_kv_heads, neuron_indices]
    v_weights = v_weights[head_indices][:, :new_num_kv_heads, neuron_indices]
    o_weights = o_weights[neuron_indices][:, head_indices, :new_num_kv_heads]

    new_attn.q.weight.data = q_weights.reshape(-1, len(neuron_indices)).float()
    new_attn.k.weight.data = k_weights.reshape(-1, len(neuron_indices)).float()
    new_attn.v.weight.data = v_weights.reshape(-1, len(neuron_indices)).float()
    new_attn.o.weight.data = o_weights.reshape(len(neuron_indices), -1).float()

def subclone_weight(old_layer, new_layer, activations, sort_dim, pre_adjust=None):
    dim_k = new_layer.weight.data.shape[0]
    if sort_dim == 1:
        dim_x = dim_k
        dim_k = new_layer.weight.data.shape[1]
    neuron_importance = activations.abs()
    neuron_importance = neuron_importance.sum(dim=[0])
    if (neuron_importance.dim() > 1):
        neuron_importance = neuron_importance.sum(dim=[0])
    
    sorted_neurons = torch.topk(neuron_importance, k=dim_k, sorted=False).indices
    weight_data = old_layer.weight.data
    
    if sort_dim == 0:
        new_layer.weight.data = weight_data[sorted_neurons].float()
    elif sort_dim == 1:
        if pre_adjust is not None:
            new_layer.weight.data = pre_adjust(weight_data)[:dim_x, sorted_neurons].float()
        else:
            new_layer.weight.data = weight_data[:dim_x, sorted_neurons].float()
    elif sort_dim == 2:
        if pre_adjust is not None:
            new_layer.weight.data = pre_adjust(weight_data)[:, :, sorted_neurons].float()
        else:
            new_layer.weight.data = weight_data[:, :, sorted_neurons].float()

    if (weight_data.dim() > 1):
        new_layer.weight.data *= ((weight_data.shape[-1]/new_layer.weight.data.shape[-1]) ** 0.5)
    

def subclone_both_weight(old_layer, new_layer, outact, inact):
    new_input_dim = new_layer.weight.data.shape[1]
    new_output_dim = new_layer.weight.data.shape[0]
    def get_neuron_importance(activations, dim_k):
        neuron_importance = activations.abs()
        neuron_importance = neuron_importance.sum(dim=[0])
        if (neuron_importance.dim() > 1):
            neuron_importance = neuron_importance.sum(dim=[0])
        
        sorted_neurons = torch.topk(neuron_importance, k=dim_k, sorted=False).indices
        return sorted_neurons

    out_importance = get_neuron_importance(outact, new_output_dim)
    in_importance = get_neuron_importance(inact, new_input_dim)
    
    new_layer.weight.data = old_layer.weight.data[out_importance][:, in_importance].float() 
    if (new_layer.weight.data.dim() > 1):
        new_layer.weight.data *= ((old_layer.weight.data.shape[-1]/new_layer.weight.data.shape[-1]) ** 0.5)
    

def subclone_layer(old_idx, activations,old_layer, new_layer, neuron_indices, head_indices, new_model, is_decoder=False):
    subclone_attention(old_layer.layer[0].SelfAttention, new_layer.layer[0].SelfAttention, neuron_indices, head_indices)
    if is_decoder:
        subclone_attention(old_layer.layer[1].EncDecAttention, new_layer.layer[1].EncDecAttention, neuron_indices, head_indices)

    new_layer_named_modules = dict(new_layer.layer.named_modules())

    def pre_adjust_setup(name):
        if "down_proj" in name:
            return lambda x: x[neuron_indices, :]
        elif "gate_proj" in name:
            return lambda x: x[:, neuron_indices]
        elif "SelfAttention.o" in name:
            return lambda x: x[:, neuron_indices]
        elif "norm" in name:
            return lambda x: x[neuron_indices]
        return None
    
    for name, module in old_layer.layer.named_modules():
        if len(name) == 0:
            continue
        if not ((isinstance(module, torch.nn.Linear)) or ("norm" in name)):
            continue

        new_module = new_layer_named_modules[name]
        
        if (f"{old_idx}_{name}" in activations["out"] and f"{old_idx}_{name}" in activations["in"]):

            #print(f"In/out cloning: {old_idx}_{name}")
            outact = activations["out"][f"{old_idx}_{name}"]
            inact = activations["in"][f"{old_idx}_{name}"]
            subclone_both_weight(module, new_module, outact, inact)

        elif (f"{old_idx}_{name}" in activations["out"]):
      
            #print(f"Out cloning: {old_idx}_{name}")
            outact = activations["out"][f"{old_idx}_{name}"]
            subclone_weight(module, new_module, outact, 0, pre_adjust=pre_adjust_setup(name))

        elif (f"{old_idx}_{name}" in activations["in"]):
      
            #print(f"In cloning: {old_idx}_{name}")
            inact = activations["in"][f"{old_idx}_{name}"]
            subclone_weight(module, new_module, inact, 1, pre_adjust=pre_adjust_setup(name))  
    

def subclone_model(base_model, new_model, activations_encoder, activations_decoder, activations_shared):
    if ("subcloned_model" in locals()):
        del subcloned_model

    global_neuron_importance_encoder = compute_global_neuron_importance(activations_encoder)
    global_neuron_importance_decoder = compute_global_neuron_importance(activations_decoder)

    head_importance_encoder = compute_head_importance(activations_encoder, base_model.config.num_attention_heads)
    head_importance_decoder = compute_head_importance(activations_decoder, base_model.config.num_attention_heads)
    
    # Select top neurons
    top_neurons_encoder = torch.topk(global_neuron_importance_encoder, k=new_model.config.hidden_size, sorted=False).indices # global
    top_neurons_decoder = torch.topk(global_neuron_importance_decoder, k=new_model.config.hidden_size, sorted=False).indices # global
    # Select top heads
    top_heads_encoder = torch.topk(head_importance_encoder, k=new_model.config.num_attention_heads, sorted=False).indices # for each layer (layer, head)
    top_heads_decoder = torch.topk(head_importance_decoder, k=new_model.config.num_attention_heads, sorted=False).indices # for each layer (layer, head)
    
    # Subclone embedding
    subclone_weight(base_model.shared, new_model.shared, activations_shared["main"]["embed"], 1)
    subclone_weight(base_model.lm_head, new_model.lm_head, activations_shared["main"]["unembed"], 1)
    subclone_weight(base_model.encoder.final_layer_norm, new_model.encoder.final_layer_norm, activations_encoder["main"]["norm"], 0)
    subclone_weight(base_model.decoder.final_layer_norm, new_model.decoder.final_layer_norm, activations_decoder["main"]["norm"], 0)

    subclone_weight(base_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias, new_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias, activations_encoder["main"]["bias"], 1)
    subclone_weight(base_model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias, new_model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias, activations_decoder["main"]["bias"], 1)
    
    # Subclone layers
    encoder_layers_old = base_model.config.num_layers
    decoder_layers_old = base_model.config.num_decoder_layers
    encoder_layers_new = new_model.config.num_layers
    decoder_layers_new = new_model.config.num_decoder_layers

    # Take half of the layers from the beginning and the end
    encoder_layers_to_keep = list(range(math.ceil(encoder_layers_new / 2))) + list(range(-math.floor(encoder_layers_new/2), 0))
    decoder_layers_to_keep = list(range(math.ceil(decoder_layers_new / 2))) + list(range(-math.floor(decoder_layers_new/2), 0))

    for new_idx, old_idx in enumerate(encoder_layers_to_keep):
        if old_idx < 0:
            old_idx = len(base_model.encoder.block) + old_idx
        subclone_layer(old_idx, activations_encoder ,base_model.encoder.block[old_idx], new_model.encoder.block[new_idx], top_neurons_encoder, top_heads_encoder[old_idx], new_model)

    for new_idx, old_idx in enumerate(decoder_layers_to_keep):
        if old_idx < 0:
            old_idx = len(base_model.decoder.block) + old_idx
        subclone_layer(old_idx, activations_decoder ,base_model.decoder.block[old_idx], new_model.decoder.block[new_idx], top_neurons_decoder, top_heads_decoder[old_idx], new_model)
    
    return new_model

class WeightCloner:
    def __init__(self, teacher_model, dataset: OnlineDistilledDataset, tokenizer):
        self.teacher_model = teacher_model
        self.activations_encoder, self.activations_decoder, self.activations_shared = collect_activations(self.teacher_model, dataset, tokenizer)

    def clone_weights(self, student_model):
        """
        Clones the weights from the teacher model to the student model.
        """
        student_model = subclone_model(self.teacher_model, student_model, self.activations_encoder, self.activations_decoder, self.activations_shared)
        return student_model