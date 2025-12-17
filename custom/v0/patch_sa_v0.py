# patch_sa_v0.py
import torch

# 你的 v0 wrapper：from sa import sa_forward
# 這裡假設你已經能 import sa_forward（sa.py + sa_cuda_ext 都已編好）
from sa import sa_forward


def patch_vit_pytorch_attention_sa_v0(model):
    """
    Replace vit-pytorch Attention core with your v0 CUDA SA kernel.
    Keep to_qkv / to_out same, only change attention computation.

    Expect Attention module has:
      - self.to_qkv
      - self.to_out
      - self.heads
      - (optional) self.scale
    """
    patched = 0

    for name, m in model.named_modules():
        if m.__class__.__name__ != "Attention":
            continue

        if not (hasattr(m, "to_qkv") and hasattr(m, "to_out") and hasattr(m, "heads")):
            continue

        def forward_sa_v0(self, x):
            b, n, _ = x.shape
            h = self.heads

            # qkv: (b, n, 3 * inner_dim)
            qkv = self.to_qkv(x).chunk(3, dim=-1)

            # reshape to (b, h, n, d)
            q, k, v = [t.view(b, n, h, -1).transpose(1, 2).contiguous() for t in qkv]

            d = q.size(-1)
            scale = float(getattr(self, "scale", d ** -0.5))

            # v0 kernel expects fp16 contiguous CUDA
            if q.dtype != torch.float16:
                q = q.half()
                k = k.half()
                v = v.half()

            # SA core: O = softmax(QK^T * scale) V
            out = sa_forward(q, k, v, scale)  # (b, h, n, d)

            # back to (b, n, h*d)
            out = out.transpose(1, 2).reshape(b, n, -1).contiguous()
            out = out.float()

            return self.to_out(out)

        # bind patched forward
        m.forward = forward_sa_v0.__get__(m, m.__class__)
        patched += 1
        print(f"[INFO] Patched Attention with SA v0: {name}")

    print(f"[INFO] Total patched Attention modules: {patched}")
    return model
