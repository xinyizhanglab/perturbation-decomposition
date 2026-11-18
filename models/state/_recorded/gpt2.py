"""Recorded arc-state 0.10.2 bidirectional GPT2 wrapper."""
import torch
from transformers import GPT2Config, GPT2Model

class GPT2BidirectionalModel(GPT2Model):
    """
    A thin wrapper around GPT2Model that disables the causal (unidirectional) mask,
    allowing full bidirectional attention—and prints the internal bias mask each forward pass.
    """

    def __init__(self, config: GPT2Config):
        # Mark as not‐a‐decoder (for downstream utilities).
        config.is_decoder = False
        super().__init__(config)

        # Overwrite each attention's bias so no triangular masking occurs.
        for block in self.h:
            # block.attn.bias is a bool‐tensor of shape (1, 1, max_pos, max_pos).
            if hasattr(block.attn, "bias") and block.attn.bias is not None:
                block.attn.bias.data.fill_(True)
            block.attn.is_causal = False

        def _no_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            return None

        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Determine sequence length for printing the relevant slice of bias
        if input_ids is not None:
            seq_len = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.size(1)
        else:
            seq_len = None  # If neither is given, we can’t infer seq_len

        if seq_len is not None:
            # Print the (1, 1, seq_len, seq_len) slice of the bias for the first block
            if hasattr(self.h[0].attn, "bias") and self.h[0].attn.bias is not None:
                bias_mask = self.h[0].attn.bias[0, 0, :seq_len, :seq_len]
            else:
                bias_mask = None
        #     print("Bias mask (block 0) slice [0,0,:seq_len,:seq_len]:")
        #     print(bias_mask)
        # else:
        #     print("Cannot infer sequence length to print bias mask.")

        # If a 2D attention_mask was provided, print its expanded 4D version:
        if attention_mask is not None:
            # Expand to (batch_size, 1, seq_len, seq_len)
            B, S = attention_mask.size()
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
            # Convert to float mask (1→0.0, 0→-inf) just like GPT2 does internally
            neg_inf = torch.finfo(self.dtype).min
            float_mask = (1.0 - expanded.to(self.dtype)) * neg_inf
            # print(f"Expanded attention_mask (shape {expanded.shape}) → float mask:")
            # print(float_mask)

        # Finally, call the parent forward method
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
