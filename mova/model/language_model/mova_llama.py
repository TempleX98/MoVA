#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..arch import MovaMetaModel, MovaMetaForCausalLM


class MovaConfig(LlamaConfig):
    model_type = "mova_llama"


class MovaLlamaModel(MovaMetaModel, LlamaModel):
    config_class = MovaConfig

    def __init__(self, config: LlamaConfig):
        super(MovaLlamaModel, self).__init__(config)


class MovaLlamaForCausalLM(LlamaForCausalLM, MovaMetaForCausalLM):
    config_class = MovaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MovaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        high_images: Optional[torch.FloatTensor] = None,
        flattened_patches: Optional[torch.FloatTensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        cached_features: Optional[torch.Tensor] = None,
        prompts: Optional[str] = None,
        has_routing: Optional[bool] = None,        
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images, 
                high_images, 
                flattened_patches, 
                routing_weights, 
                cached_features,
                prompts, 
                has_routing,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        high_images: Optional[torch.FloatTensor] = None,
        flattened_patches: Optional[torch.FloatTensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        cached_features: Optional[torch.Tensor] = None,
        prompts: Optional[str] = None,
        has_routing: Optional[bool] = None,        
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                high_images, 
                flattened_patches, 
                routing_weights, 
                cached_features,
                prompts, 
                has_routing,                
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        high_images = kwargs.pop("high_images", None)
        flattened_patches = kwargs.pop("flattened_patches", None)
        routing_weights = kwargs.pop("routing_weights", None)
        cached_features = kwargs.pop("cached_features", None)
        prompts = kwargs.pop("prompts", None)
        has_routing = kwargs.pop("has_routing", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if high_images is not None:
            inputs['high_images'] = high_images
        if flattened_patches is not None:
            inputs['flattened_patches'] = flattened_patches
        if routing_weights is not None:
            inputs['routing_weights'] = routing_weights
        if cached_features is not None:
            inputs['cached_features'] = cached_features
        if prompts is not None:
            inputs['prompts'] = prompts
        if has_routing is not None:
            inputs['has_routing'] = has_routing 
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("mova_llama", MovaConfig)
AutoModelForCausalLM.register(MovaConfig, MovaLlamaForCausalLM)
