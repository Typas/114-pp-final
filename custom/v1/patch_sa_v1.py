import torch
from sa_v1 import sa_forward_v1

def patch_vit_pytorch_attention_sa_v1(model):
    patched = 0
    for name, m in model.named_modules():
        if m.__class__.__name__ != "Attention":
            continue
        if not (hasattr(m, "to_qkv") and hasattr(m, "to_out") and hasattr(m, "heads")):
            continue

        def forward_sa_v1(self, x):
            b, n, _ = x.shape
            h = self.heads
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = [t.view(b, n, h, -1).transpose(1, 2).contiguous() for t in qkv]

            d = q.size(-1)
            scale = float(getattr(self, "scale", d ** -0.5))

            # v1 kernel expects fp16
            if q.dtype != torch.float16:
                q = q.half(); k = k.half(); v = v.half()

            out = sa_forward_v1(q.contiguous(), k.contiguous(), v.contiguous(), scale)  # (b,h,n,d)
            out = out.transpose(1, 2).reshape(b, n, -1).contiguous()

            # keep vit-pytorch linear weights fp32 → cast back
            out = out.float()
            return self.to_out(out)

        m.forward = forward_sa_v1.__get__(m, m.__class__)
        patched += 1
        print(f"[INFO] Patched Attention with SA v1: {name}")

    print(f"[INFO] Total patched Attention modules: {patched}")
    return model
