# MoVA: Adapting Mixture of Vision Experts to Multimodal Context

Official repository for the paper "MoVA: Adapting Mixture of Vision Experts to Multimodal Context".

[[📖 Paper](https://arxiv.org/pdf/2404.13046)] 


## 💥 News

- **[2024.04.22]** 🚀 We release our paper on arXiv.

## 📌 TODO
- [ ] Release training code and models.

## 👀 About MoVA

To alleviate the bias of CLIP vision encoder, we first delve into the inherent behavior of different pre-trained vision encoders and then propose the **MoVA**, a powerful and novel MLLM, adaptively routing and fusing task-specific vision experts with a coarse-to-fine mechanism.

![demo](figures/framework.png)

MoVA consists of two stages: coarse-grained context-ware expert routing and fine-grained expert fusion with MoV-Adapter.
1. **Coarse-grained context-ware expert routing**: 
First, MoVA leverages the tool-use capabilities of LLM, routing the most appropriate experts from $N$ expert candidates via LLM to help the model answer the user's question. We incorporate the expert-routing LoRA module into the LLM to improve the efficiency and effectiveness of expert routing.
This expert-routing LoRA module is trained with expert routing annotations and can better align the LLM and the routing task.
2. **Fine-grained expert fusion with MoV-Adapter**: 
In the second stage, we turn to enhance the visual representation with a novel MoV-Adapter module in a fine-grained manner.
More specifically, we leverage the cross-attention mechanism to extract the task-specific knowledge of representations from chosen experts.
Meanwhile, the dynamic gating network in MoV-Adapter can allocate soft weights to the extracted knowledge of each expert according to the input image and instruction.
Then the extracted knowledge can be effectively integrated into the foundational representation of the base vision encoder.

MoVA with **Vicuna-7B** and **Hermes-Yi-34B** can achieve significant performance gains over current state-of-the-art methods in a wide range of challenging benchmarks.

## 🧠 Acknowledgement

We would like to thank the following repos for their great work:

- The codebase of MoVA is built upon [LLaVA](https://github.com/haotian-liu/LLaVA).
- MoVA incorporates vision encoders from [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), [Co-DETR](https://github.com/Sense-X/Co-DETR), [SAM](https://github.com/facebookresearch/segment-anything), [Pix2Struct](https://github.com/google-research/pix2struct), [Deplot](https://huggingface.co/google/deplot), [Vary](https://github.com/Ucas-HaoranWei/Vary/tree/main), and [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

## ✅ Citation

If you find **MoVA** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{zong2024mova,
  title={MoVA: Adapting Mixture of Vision Experts to Multimodal Context},
  author={Zong, Zhuofan and Ma, Bingqi and Shen, Dazhong and Song, Guanglu and Shao, Hao and Jiang, Dongzhi and Li, Hongsheng and Liu, Yu},
  journal={arXiv preprint arXiv:2404.13046},
  year={2024}
}
```