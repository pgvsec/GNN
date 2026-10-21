"""Write the Layer I backbone stored inside a Layer III checkpoint as a standalone Layer I
checkpoint (cfg / model / ema), so it can be passed as --l1_ckpt.

Before the loader fix (composition_eval.load_models, assemble.load_l3_model), loading an L3
checkpoint overwrote the given L1 weights with this backbone, so every pipeline number from
before the fix was produced with it. Use this to reproduce those numbers with the fixed code.

  python extract_backbone.py --l1_ckpt .../layer1_v10/best.pt \\
      --l3_ckpt .../layer3_v12/best.pt --out .../layer1_from_l3v12.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1_ckpt", required=True, help="any Layer I checkpoint with the same cfg (for cfg only)")
    ap.add_argument("--l3_ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck1 = torch.load(args.l1_ckpt, map_location="cpu", weights_only=False)
    ck3 = torch.load(args.l3_ckpt, map_location="cpu", weights_only=False)
    sd3 = ck3.get("ema", ck3.get("model"))
    sd = {k[len("backbone."):]: v for k, v in sd3.items() if k.startswith("backbone.")}
    ref = ck1.get("ema", ck1.get("model"))
    missing = sorted(set(ref) - set(sd))
    if missing:
        raise SystemExit(f"L3 checkpoint lacks backbone weights for {len(missing)} keys, e.g. {missing[:3]}")
    diff = max(float((sd[k].float() - ref[k].float()).abs().max()) for k in ref if sd[k].shape == ref[k].shape)
    torch.save({"cfg": ck1["cfg"], "model": sd, "ema": sd}, args.out)
    print(f"wrote {args.out} ({len(sd)} tensors); max |diff| to {args.l1_ckpt}: {diff:.3g}")


if __name__ == "__main__":
    main()
