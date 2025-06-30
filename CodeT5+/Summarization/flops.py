"""Computes the flops needed for training/running transformer networks."""
"""Partial code is from https://github.com/google-research/electra/blob/master/flops_computation.py"""

import collections

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class TransformerHparams(object):
    """Computes the train/inference FLOPs for transformers."""

    def __init__(self, h=768, l=12, s=514, v=50265, i=3072, heads=12, head_size=None, output_frac=0.15625, sparse_embed_lookup=False, decoder=False):
        self.h = h  # hidden size
        self.l = l  # number of layers
        self.s = s  # sequence length, clone detection needs a double sequence length
        self.v = v  # vocab size
        self.e = h  # embedding size
        self.i = h * 4 if i is None else i  # intermediate size
        self.kqv = h if head_size is None else head_size * heads
        self.heads = heads
        self.output_frac = output_frac
        self.sparse_embed_lookup = sparse_embed_lookup  # whether to use sparse embedding lookup
        self.decoder = decoder  # whether this is a decoder transformer

    def get_block_flops(self):
        attn_mult = 2 if self.decoder else 1
        block_flops = dict(
            kqv=3 * 2 * self.h * self.kqv * attn_mult,
            kqv_bias=3 * self.kqv * attn_mult,
            attention_scores=2 * self.kqv * self.s * attn_mult,
            attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * attn_mult,
            attention_dropout=DROPOUT_FLOPS * self.s * self.heads * attn_mult,
            attention_scale=self.s * self.heads * attn_mult,
            attention_weighted_avg_values=2 * self.h * self.s * attn_mult,
            attn_output=2 * self.h * self.h * attn_mult,
            attn_output_bias=self.h * attn_mult,
            attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mult,
            attn_output_residual=self.h * attn_mult,
            attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mult,
            intermediate=2 * self.h * self.i,
            intermediate_act=ACTIVATION_FLOPS * self.i,
            intermediate_bias=self.i,
            output=2 * self.h * self.i,
            output_bias=self.h,
            output_dropout=DROPOUT_FLOPS * self.h,
            output_residual=self.h,
            output_layer_norm=LAYER_NORM_FLOPS * self.h
        )
        return sum(block_flops.values()) * self.s

    def get_embedding_flops(self, output=False):
        """Get the forward-pass FLOPs the transformer inputs or output softmax."""
        embedding_flops = {}
        if output or (not self.sparse_embed_lookup):
            embedding_flops["main_multiply"] = 2 * self.e * self.v
        # input embedding post-processing
        if not output:
            embedding_flops.update(dict(
                tok_type_and_position=2 * self.e * (self.s + 2),
                add_tok_type_and_position=2 * self.e,
                emb_layer_norm=LAYER_NORM_FLOPS * self.e,
                emb_dropout=DROPOUT_FLOPS * self.e
            ))
        # projection layer if e != h
        if self.e != self.h or output:
            embedding_flops.update(dict(
                hidden_kernel=2 * self.h * self.e,
                hidden_bias=self.e if output else self.h
            ))
        # extra hidden layer and output softmax
            if output:
                embedding_flops.update(dict(
                    hidden_activation=ACTIVATION_FLOPS * self.e,
                    hidden_layernorm=LAYER_NORM_FLOPS * self.e,
                    output_softmax=SOFTMAX_FLOPS * self.v,
                    output_target_word=2 * self.v
                ))
                return self.output_frac * sum(embedding_flops.values()) * self.s
        return sum(embedding_flops.values()) * self.s

    def get_binary_classification_flops(self):
        classification_flops = dict(
            hidden=2 * self.h * self.h,
            hidden_bias=self.h,
            hidden_act=DROPOUT_FLOPS * self.h + ACTIVATION_FLOPS * self.h,
            logits=2 * self.h
            # soft_logits=2 * SOFTMAX_FLOPS
        )
        return sum(classification_flops.values()) * self.s
    
    def get_generation_flops(self):
        generation_flops = dict(
            hidden=2 * self.h * self.h,
            hidden_bias=self.h,
            hidden_act=DROPOUT_FLOPS * self.h + ACTIVATION_FLOPS * self.h,
            logits=self.v * self.h
        )
        return sum(classification_flops.values()) * self.s

    def get_infer_flops(self):
        """Get the FLOPs for running inference with the transformer on a
        classification task."""
        # return (self.get_embedding_flops())
        return ((self.l * self.get_block_flops()) +
                self.get_embedding_flops() +
                self.get_binary_classification_flops())

    def get_params(self):
        embedding_params = {}
        embedding_params.update(dict(
            token_params=self.v * self.h,
            position_params=self.s * self.h,
            type_and_layer_norm=self.h * 3
        ))

        block_params = {}
        block_params.update(dict(
            attention_params=3 * (self.h * self.h + self.h),
            linear_params=self.h * self.h + self.h,
            fnn_params=self.h * self.i * 2 + self.i + self.h,
            layer_norm=self.h * 4,
            # pooler_params=self.h*self.h + self.h
        ))

        classification_params = {}
        classification_params.update(dict(
            pooler_params=self.h*self.h + self.h,
            dense_params=self.h * self.h + self.h,
            linear_params=self.h * 2 + 2
        ))

        return sum(embedding_params.values()) + sum(block_params.values()) * self.l + sum(classification_params.values())

    def non_embedding_params(self):
        block_params = {}
        block_params.update(dict(
            attention_params=3 * (self.h * self.h + self.h),
            linear_params=self.h * self.h + self.h,
            fnn_params=self.h * self.i * 2 + self.i + self.h,
            layer_norm=self.h * 4,
            # pooler_params=self.h*self.h + self.h
        ))

        # classification_params = {}
        # classification_params.update(dict(
        #     pooler_params=self.h*self.h + self.h,
        #     dense_params=self.h * self.h + self.h,
        #     linear_params=self.h * 2 + 2
        # ))

        return sum(block_params.values()) * self.l #+ sum(classification_params.values())


class EncoderDecoderHparams(object):
    """Computes the train/inference FLOPs for encoder-decoder transformers."""

    def __init__(self, h=768, l_encoder=12, l_decoder=12, s=514, v=50265, i=3072, heads=12, head_size=None, gen_len=128):
        self.encoder = TransformerHparams(h, l_encoder, s=s, v=v, i=i, heads=heads, head_size=head_size, output_frac=0)
        self.decoder = TransformerHparams(h, l_decoder, s=s, v=v, i=i, heads=heads, head_size=head_size, decoder=True, output_frac=1)
        self.gen_len = gen_len  # generation length
    
    def get_params(self):
        return self.encoder.get_params() + self.decoder.get_params()

    def get_infer_flops(self):
        encoder_flops = self.encoder.get_embedding_flops() + self.encoder.l * self.encoder.get_block_flops()
        decoder_flops = self.decoder.get_embedding_flops() + self.decoder.get_embedding_flops(output=True) + self.decoder.l * self.decoder.get_block_flops()

        return encoder_flops + decoder_flops * self.gen_len

MODEL_FLOPS = collections.OrderedDict([
    ("roberta", [TransformerHparams().get_infer_flops(),
     TransformerHparams().get_params()])
])

def main():
    model = TransformerHparams(768, 12, 256, 50265, 3072, 12)
    flops = model.get_infer_flops()
    params = model.get_params()
    print(flops, params)

    model = TransformerHparams(768, 12, 514, 50265, 3072, 12)
    flops = model.get_infer_flops()
    params = model.get_params()
    print(flops/1e9)

    model = TransformerHparams(24, 1, 514, 27505, 1508, 1)
    flops = model.get_infer_flops()
    params = model.get_params()
    print(flops/1e9)

    model = TransformerHparams(96, 12, 514, 1000, 64, 8)
    flops = model.get_infer_flops()
    params = model.get_params()
    print(flops/1e9)


    model = EncoderDecoderHparams(483, 5, 11, 256, 50265, 1596, 7, 19)
    flops = model.get_infer_flops()
    params = model.get_params()
    print(flops/1e9)
    print(params*4/1e6)  # in MB

if __name__ == "__main__":
    main()
