import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionConfig,
    AutoImageProcessor,
    Pix2StructForConditionalGeneration,
    Pix2StructImageProcessor
)
from mova.model.vision_experts.dinov2.modeling_dinov2 import Dinov2Model
from mova.model.vision_experts.vary.sam import build_sam_vit_b, build_sam_vit_h
from mova.model.vision_experts.codetr.vit import build_codetr_vit_l
import open_clip
import os


def resize_image_embeddings(model, size):
    num_patches = size // model.vision_model.embeddings.patch_size
    embedding = model.vision_model.embeddings.position_embedding.weight.data
    cls_embed = embedding[:1, :]
    pos_embed = embedding[1:, :].reshape(
        1, int(model.vision_model.embeddings.num_patches**0.5), int(model.vision_model.embeddings.num_patches**0.5), -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed, size=(num_patches, num_patches), mode='bicubic', align_corners=False). \
        reshape(1, -1, num_patches**2).permute(0, 2, 1)
    pos_embed = torch.cat([cls_embed, pos_embed[0]], dim=0)
    model.vision_model.embeddings.position_embedding.weight.data = pos_embed
    model.vision_model.embeddings.position_ids.data = torch.arange(num_patches**2+1).expand((1, -1))
    return model


class Pix2StructImageProcessorWarp(Pix2StructImageProcessor):
    def preprocess(self, *args, **kwargs):
        result = super().preprocess(*args, **kwargs)
        result["pixel_values"] = result["flattened_patches"]
        return result

class MoVAVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        assert isinstance(vision_tower, list)
        self.vision_tower_names = vision_tower
        # vision experts should be in order: clip, dinov2, vary, codetr, pix2struct, deplot, biomedclip, sam
        # vision channels: 1024, 1536, 512, 256, 1536, 768, 768, 256
        self.select_layer = args.mm_vision_select_layer
        self.ft_vision_tower = getattr(args, "ft_vision_tower", False)
        self.ft_vision_tower_last_n_layer = getattr(
            args, "ft_vision_tower_last_n_layer", -1
        )
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.image_size = args.image_feat_size * 14        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only_0 = CLIPVisionConfig.from_pretrained(self.vision_tower_names[0])
        self.out_channels = args.expert_channels

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor_0 = CLIPImageProcessor.from_pretrained(
            self.vision_tower_names[0]
        )
        self.image_processor_0.size['shortest_edge'] = self.image_size
        self.image_processor_0.crop_size['height'] = self.image_size
        self.image_processor_0.crop_size['width'] = self.image_size
        self.vision_tower_0 = CLIPVisionModel.from_pretrained(self.vision_tower_names[0])
        self.vision_tower_0 = resize_image_embeddings(self.vision_tower_0, self.image_size)

        self.image_processor_1 = AutoImageProcessor.from_pretrained(
            self.vision_tower_names[1]
        )
        self.image_processor_1.size['shortest_edge'] = self.image_size
        self.image_processor_1.crop_size['height'] = self.image_size
        self.image_processor_1.crop_size['width'] = self.image_size        
        self.vision_tower_1 = Dinov2Model.from_pretrained(self.vision_tower_names[1])

        self.image_processor_2 = AutoImageProcessor.from_pretrained(
            self.vision_tower_names[2]
        )
        self.vision_tower_2 = build_sam_vit_b(checkpoint=os.path.join(self.vision_tower_names[2], 'pytorch_model.bin') if os.path.exists(os.path.join(self.vision_tower_names[2], 'pytorch_model.bin')) else None)

        self.vision_tower_3 = build_codetr_vit_l(checkpoint=self.vision_tower_names[3] if os.path.exists(self.vision_tower_names[3]) else None)

        self.vision_tower_4 = Pix2StructForConditionalGeneration.from_pretrained(self.vision_tower_names[4]).get_encoder()
        self.image_processor_4 = Pix2StructImageProcessorWarp.from_pretrained(self.vision_tower_names[4])
        self.image_processor_4.is_vqa = False
        self.image_processor_4.crop_size = {"height": 45, "width": 45}
        self.image_processor_4.image_mean = [0.48145466, 0.4578275, 0.40821073]


        self.vision_tower_5 = Pix2StructForConditionalGeneration.from_pretrained(self.vision_tower_names[5]).get_encoder()

        # self.vision_tower_6, _ = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.vision_tower_6, _ = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir=self.vision_tower_names[6])
        self.vision_tower_6 = self.vision_tower_6.visual

        self.vision_tower_7 =  build_sam_vit_h(checkpoint=self.vision_tower_names[7] if os.path.exists(self.vision_tower_names[2]) else None)

        if not self.ft_vision_tower:
            self.vision_tower_0.requires_grad_(False)
        else:
            if self.ft_vision_tower_last_n_layer == -1:
                self.vision_tower_0.requires_grad_(True)
            else:
                self.vision_tower_0.requires_grad_(False)
                layer_num = self.vision_tower_0.config.num_hidden_layers
                start = layer_num - self.ft_vision_tower_last_n_layer
                for i in range(start, layer_num):
                    self.vision_tower_0.vision_model.encoder.layers[i].requires_grad_(True)

        self.vision_tower_1.requires_grad_(False)
        self.vision_tower_2.requires_grad_(False)
        self.vision_tower_3.requires_grad_(False)
        self.vision_tower_4.requires_grad_(False)
        self.vision_tower_5.requires_grad_(False)
        self.vision_tower_6.requires_grad_(False)
        self.vision_tower_7.requires_grad_(False)

        self.is_loaded = True

    def set_training_mode(self):
        if not self.ft_vision_tower:
            self.vision_tower_0.requires_grad_(False)
        else:
            if self.ft_vision_tower_last_n_layer == -1:
                self.vision_tower_0.requires_grad_(True)
            else:
                self.vision_tower_0.requires_grad_(False)
                layer_num = self.vision_tower_0.config.num_hidden_layers
                start = layer_num - self.ft_vision_tower_last_n_layer
                for i in range(start, layer_num):
                    self.vision_tower_0.vision_model.encoder.layers[i].requires_grad_(
                        True
                    )
        self.vision_tower_1.requires_grad_(False)
        self.vision_tower_2.requires_grad_(False)
        self.vision_tower_3.requires_grad_(False)
        self.vision_tower_4.requires_grad_(False)
        self.vision_tower_5.requires_grad_(False)
        self.vision_tower_6.requires_grad_(False)
        self.vision_tower_7.requires_grad_(False)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def feature_align(self, feature, size):
        if feature.shape[1] != size:
            new_size = (int(size**0.5), int(size**0.5))
            B, L, C = feature.shape
            size = (int(L**0.5), int(L**0.5))
            feature = feature.permute(0, 2, 1).reshape(B, C, size[0], size[1]).contiguous()
            new_feature = F.interpolate(feature.float(), size=new_size, mode="bilinear", align_corners=False)
            new_feature = new_feature.reshape(B, C, -1).permute(0, 2, 1).contiguous()
            return new_feature.to(feature.dtype)
        else:
            return feature

    def merge_feature(self, base_feat, expert_feat):
        expert_feat = expert_feat.to(base_feat.dtype)
        expert_feat = self.feature_align(expert_feat, base_feat.shape[1])
        base_feat = torch.cat((base_feat, expert_feat), dim=-1)
        return base_feat

    def forward(self, images, high_images=None, flattened_patches=None, routing_weights=None, cached_features=None):
        if self.ft_vision_tower:
            return self.forward_func(images, high_images, flattened_patches, routing_weights, cached_features)
        else:
            with torch.no_grad():
                return self.forward_func(images, high_images, flattened_patches, routing_weights, cached_features)

    def forward_func(self, images, high_images, flattened_patches, routing_weights, cached_features):
        if type(images) is list:
            raise NotImplementedError
            image_features = []
            for image, high_image, flattened_patch in zip(images, high_images, flattened_patches):
                image_forward_out_0 = self.vision_tower_0(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                with torch.no_grad():
                    if image.shape[-1] > 518:
                        tmp_image = image.to(device=self.device, dtype=self.dtype)
                        tmp_image = F.interpolate(tmp_image.float(), size=(518, 518), mode="bilinear").to(image.dtype)
                        images_1 = tmp_image
                    else:
                        images_1 = image
                    image_forward_out_1 = self.vision_tower_1(
                        images_1.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    )

                feature0 = self.feature_select(image_forward_out_0).to(image.dtype)
                feature1 = self.feature_select(image_forward_out_1).to(image.dtype)

                image_feature = self.merge_feature(image_feature, feature1)

                with torch.no_grad():
                    image_forward_out_2 = self.vision_tower_2(
                        high_image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                    )
                image_feature = self.merge_feature(image_feature, image_forward_out_2)

                with torch.no_grad():
                    image_forward_out_3 = self.vision_tower_3(
                        high_image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                    )
                image_feature = self.merge_feature(image_feature, image_forward_out_3)

                with torch.no_grad():
                    image_forward_out_4 = self.vision_tower_4(
                        flattened_patch.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    ).last_hidden_state
                image_feature = self.merge_feature(image_feature, image_forward_out_4)

                with torch.no_grad():
                    image_forward_out_5 = self.vision_tower_5(
                        flattened_patch.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                    ).last_hidden_state
                image_feature = self.merge_feature(image_feature, image_forward_out_5)

                with torch.no_grad():
                    tmp_image = image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                    tmp_image = F.interpolate(tmp_image.float(), size=(224, 224), mode="bilinear").to(image.dtype)
                    image_forward_out_6 = self.vision_tower_6(
                        tmp_image
                    )[:, 1:]
                image_feature = self.merge_feature(image_feature, image_forward_out_6)

                with torch.no_grad():
                    image_forward_out_7 = self.vision_tower_7(
                        high_image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                    )
                image_feature = self.merge_feature(image_feature, image_forward_out_7)

                image_features.append(image_feature)
        else:
            new_cached_features = []
            _, _, h, w = images.shape
            images = images.reshape(-1, 2, 3, h, w).transpose(0, 1)
            images_0, images_1 = images[0], images[1]

            # CLIP base encoder
            if cached_features is not None and cached_features[0] is not None:
                feature0 = cached_features[0]
            else:
                image_forward_outs_0 = self.vision_tower_0(
                    images_0.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                )
                feature0 = self.feature_select(image_forward_outs_0).to(images.dtype)
            new_cached_features.append(feature0)

            # DINOv2
            if cached_features is not None and cached_features[1] is not None:
                feature1 = cached_features[1]
                new_cached_features.append(feature1)
            elif not self.training and feature0.shape[0]==1 and routing_weights[0][0]==0:
                # Inference + batch 1 + non-relevant
                feature1 = torch.zeros_like(feature0).repeat(1, 1, 2)[:, :, :self.out_channels[0]]
                new_cached_features.append(None)
            else:
                with torch.no_grad():
                    if images_1.shape[-1] > 518:
                        tmp_image = images_1.to(device=self.device, dtype=self.dtype)
                        tmp_image = F.interpolate(tmp_image.float(), size=(518, 518), mode="bilinear").to(images.dtype)
                        images_1 = tmp_image
                    image_forward_outs_1 = self.vision_tower_1(
                        images_1.to(device=self.device, dtype=self.dtype),
                        output_hidden_states=True,
                    )
                feature1 = self.feature_select(image_forward_outs_1).to(images.dtype)
                feature1 = self.feature_align(feature1, feature0.shape[1])
                new_cached_features.append(feature1)
            image_features = torch.cat((feature0, feature1), dim=-1)

            # Vary
            images_2 = high_images
            if cached_features is not None and cached_features[2] is not None:
                image_forward_out_2 = cached_features[2]
                new_cached_features.append(image_forward_out_2)
            elif not self.training and feature0.shape[0]==1 and routing_weights[0][1]==0:
                # Inference + batch 1 + non-relevant
                image_forward_out_2 = torch.zeros_like(feature0)[:, :, :self.out_channels[1]]
                new_cached_features.append(None)
            else:
                with torch.no_grad():
                    image_forward_out_2 = self.vision_tower_2(
                        images_2.to(device=self.device, dtype=self.dtype)
                    )
                new_cached_features.append(image_forward_out_2)
            image_features = self.merge_feature(image_features, image_forward_out_2)

            # CoDETR
            images_3 = high_images
            if cached_features is not None and cached_features[3] is not None:
                image_forward_out_3 = cached_features[3]
                new_cached_features.append(image_forward_out_3)
            elif not self.training and feature0.shape[0]==1 and routing_weights[0][2]==0:
                # Inference + batch 1 + non-relevant
                image_forward_out_3 = torch.zeros_like(feature0)[:, :, :self.out_channels[2]]
                new_cached_features.append(None)
            else:
                with torch.no_grad():
                    image_forward_out_3 = self.vision_tower_3(images_3.to(device=self.device, dtype=self.dtype))
                new_cached_features.append(image_forward_out_3)
            image_features = self.merge_feature(image_features, image_forward_out_3)

            # Pix2Struct
            images_4 = flattened_patches
            if cached_features is not None and cached_features[4] is not None:
                image_forward_out_4 = cached_features[4]
                new_cached_features.append(image_forward_out_4)
            elif not self.training and feature0.shape[0]==1 and routing_weights[0][3]==0:
                # Inference + batch 1 + non-relevant
                image_forward_out_4 = torch.zeros_like(feature0).repeat(1, 1, 2)[:, :, :self.out_channels[3]]
                new_cached_features.append(None)
            else:
                with torch.no_grad():
                    image_forward_out_4 = self.vision_tower_4(
                        images_4.to(device=self.device, dtype=self.dtype),
                        output_hidden_states=True,
                    ).last_hidden_state
                new_cached_features.append(image_forward_out_4)
            image_features = self.merge_feature(image_features, image_forward_out_4)

            # Deplot
            images_5 = flattened_patches
            if cached_features is not None and cached_features[5] is not None:
                image_forward_out_5 = cached_features[5]
                new_cached_features.append(image_forward_out_5)
            elif not self.training and feature0.shape[0]==1 and routing_weights[0][4]==0:
                # Inference + batch 1 + non-relevant
                image_forward_out_5 = torch.zeros_like(feature0).repeat(1, 1, 2)[:, :, :self.out_channels[4]]
                new_cached_features.append(None)
            else:
                with torch.no_grad():
                    image_forward_out_5 = self.vision_tower_5(
                        images_5.to(device=self.device, dtype=self.dtype),
                        output_hidden_states=True,
                    ).last_hidden_state
                new_cached_features.append(image_forward_out_5)
            image_features = self.merge_feature(image_features, image_forward_out_5)

            # BiomedCLIP
            if cached_features is not None and cached_features[6] is not None:
                image_forward_out_6 = cached_features[6]
                new_cached_features.append(image_forward_out_6)
            elif not self.training and feature0.shape[0]==1 and routing_weights[0][5]==0:
                # Inference + batch 1 + non-relevant
                image_forward_out_6 = torch.zeros_like(feature0)[:, :, :self.out_channels[5]]
                new_cached_features.append(None)
            else:
                with torch.no_grad():
                    tmp_image = images_0.to(device=self.device, dtype=self.dtype)
                    tmp_image = F.interpolate(tmp_image.float(), size=(224, 224), mode="bilinear").to(images.dtype)
                    image_forward_out_6 = self.vision_tower_6(tmp_image)
                    image_forward_out_6 = image_forward_out_6[:, 1:]
                new_cached_features.append(image_forward_out_6)
            image_features = self.merge_feature(image_features, image_forward_out_6)

            # SAM
            images_7 = high_images
            if cached_features is not None and cached_features[7] is not None:
                image_forward_out_7 = cached_features[7]
                new_cached_features.append(image_forward_out_7)
            elif not self.training and feature0.shape[0]==1 and routing_weights[0][6]==0:
                # Inference + batch 1 + non-relevant
                image_forward_out_7 = torch.zeros_like(feature0)[:, :, :self.out_channels[6]]
                new_cached_features.append(None)
            else:
                with torch.no_grad():
                    image_forward_out_7 = self.vision_tower_7(
                        images_7.to(device=self.device, dtype=self.dtype)
                    )
                new_cached_features.append(image_forward_out_7)
            image_features = self.merge_feature(image_features, image_forward_out_7)

        return image_features, new_cached_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower_0.dtype

    @property
    def device(self):
        return self.vision_tower_0.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower_0.config
        else:
            return self.cfg_only_0

    @property
    def hidden_size(self):
        if self.is_loaded:
            return self.vision_tower_0.config.hidden_size
        else:
            return self.cfg_only_0.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
