"""Populate the local model caches so an OFFLINE cluster (Leonardo) can run without internet.

Run this WHERE THERE IS INTERNET (your laptop). It downloads:
  - DINOv2 backbone (torch.hub)  -> ~/.cache/torch/hub  (repo + checkpoints)
  - SegFormer detector (HF)      -> ~/.cache/huggingface
Then rsync those two cache dirs to the cluster's $TORCH_HOME / $HF_HOME (see env_leonardo.sh).

    python scripts/precache_weights.py
"""
def main():
    import torch
    print("[precache] DINOv2 backbone ...")
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    print("[precache] DINOv2 OK")
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        name = "nvidia/segformer-b3-finetuned-ade-512-512"
        print(f"[precache] SegFormer {name} ...")
        SegformerImageProcessor.from_pretrained(name)
        SegformerForSemanticSegmentation.from_pretrained(name)
        print("[precache] SegFormer OK")
    except ImportError:
        print("[precache] transformers NOT installed -> skipped SegFormer "
              "(pip install transformers, or run Track A with --doors gt which needs no detector)")
    print("[precache] done. caches: ~/.cache/torch  and  ~/.cache/huggingface")


if __name__ == "__main__":
    main()
