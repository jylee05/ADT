# src/model.py
import torch
import torch.nn as nn
import math
import random
from transformers import Wav2Vec2Model

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, condition):
        # x: (B, Seq, Dim), condition: (B, Cond_Dim)
        scale, shift = self.proj(condition).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return x * (1 + scale) + shift

class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.film1 = FiLMLayer(d_model, d_model)
        self.film2 = FiLMLayer(d_model, d_model)
        self.film3 = FiLMLayer(d_model, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.SiLU()

    def forward(self, tgt, memory, cond_emb):
        # Self Attention with FiLM
        tgt2 = self.norm1(tgt)
        tgt2 = self.film1(tgt2, cond_emb)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2)
        tgt = tgt + self.dropout(tgt2)
        
        # Cross Attention with FiLM
        tgt2 = self.norm2(tgt)
        tgt2 = self.film2(tgt2, cond_emb)
        tgt2, _ = self.cross_attn(tgt2, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        
        # FFN with FiLM
        tgt2 = self.norm3(tgt)
        tgt2 = self.film3(tgt2, cond_emb)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

class FlowMatchingTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.DRUM_CHANNELS * config.FEATURE_DIM
        
        # 1. Input Projection
        self.proj_in = nn.Linear(self.input_dim, config.HIDDEN_DIM)
        
        # 2. Time & Global Condition Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.HIDDEN_DIM),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        # 3. Condition Projections (Spectrogram & MERT)
        # Spectrogram: (B, T, 128) -> (B, T, Hidden)
        self.spec_proj = nn.Sequential(
            nn.Linear(config.N_MELS, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        # MERT: (B, T, 1024) -> (B, T, Hidden)
        self.mert_proj = nn.Sequential(
            nn.Linear(config.MERT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        # === [NEW] Learned Null Embeddings (피드백 반영) ===
        # 드롭아웃 시 0 대신 채워넣을 학습 가능한 벡터
        self.null_mert_emb = nn.Parameter(torch.randn(1, 1, config.MERT_DIM) * 0.02)
        self.null_spec_emb = nn.Parameter(torch.randn(1, 1, config.N_MELS) * 0.02)
        
        # Condition Encoders
        self.cond_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.HIDDEN_DIM, nhead=4, dim_feedforward=config.HIDDEN_DIM*2, 
                batch_first=True, norm_first=True
            ),
            num_layers=2
        )
        
        # 4. Main Decoder (FiLM Based)
        self.layers = nn.ModuleList([
            FiLMTransformerDecoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=config.N_HEADS, 
                dim_feedforward=config.HIDDEN_DIM * 4
            )
            for _ in range(config.N_LAYERS)
        ])
        
        # 5. Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, self.input_dim)
        
        # MERT Model
        print(f"Loading MERT from {config.MERT_PATH}...")
        self.mert = Wav2Vec2Model.from_pretrained(config.MERT_PATH, cache_dir=config.CACHE_DIR)
        self.mert.eval()
        for p in self.mert.parameters():
            p.requires_grad = False

    def extract_mert(self, audio):
        with torch.no_grad():
            outputs = self.mert(audio, output_hidden_states=True)
            # Use Layer 10 as per paper
            return outputs.hidden_states[self.config.MERT_LAYER_IDX]

    def apply_condition_dropout(self, feat, null_emb, drop_prob, partial_prob):
        """
        [NEW] Null Embedding과 Random Length Masking 적용
        feat: (B, T, D)
        null_emb: (1, 1, D) - Learned parameter
        """
        if not self.training:
            return feat
        
        B, T, D = feat.shape
        device = feat.device
        
        # 복사본 생성 (원본 보존)
        out_feat = feat.clone()
        
        # 1. Complete Dropout (Global)
        # 배치별로 결정: (B, 1, 1)
        drop_mask = torch.bernoulli(torch.full((B, 1, 1), drop_prob, device=device)).bool()
        
        # 마스킹 된 곳에 Null Embedding 주입
        # out_feat[drop_mask] 접근이 어려우므로 where 사용
        # drop_mask가 True(1)이면 null_emb, False(0)이면 원본
        out_feat = torch.where(drop_mask, null_emb, out_feat)
        
        # 2. Partial Dropout (Inpainting)
        # 랜덤하게 선택된 샘플에 대해 수행
        if random.random() < partial_prob:
            # [NEW] Mask Length Randomization (10% ~ 50%)
            mask_ratio = random.uniform(0.1, 0.5)
            mask_len = int(T * mask_ratio)
            
            if mask_len > 0:
                start = random.randint(0, T - mask_len)
                # 해당 구간을 Null Embedding으로 교체
                out_feat[:, start:start+mask_len, :] = null_emb

        return out_feat

    def forward(self, x_t, t, mert_feats, spec_feats):
        # x_t: (B, Seq, In_Dim)
        # t: (B,)
        
        # 1. Process Conditions with [Learned Null Embeddings]
        mert_h = self.apply_condition_dropout(
            mert_feats, self.null_mert_emb, 
            self.config.DROP_MERT_PROB, self.config.DROP_PARTIAL_PROB
        )
        spec_h = self.apply_condition_dropout(
            spec_feats, self.null_spec_emb,
            self.config.DROP_SPEC_PROB, self.config.DROP_PARTIAL_PROB
        )
        
        # Project and Encode
        mert_emb = self.mert_proj(mert_h)
        spec_emb = self.spec_proj(spec_h)
        
        # Concatenate for Cross Attention Memory
        memory = torch.cat([mert_emb, spec_emb], dim=1)
        memory = self.cond_transformer(memory)
        
        # 2. Global Condition for FiLM
        time_emb = self.time_mlp(t)
        
        # [Refined] FiLM conditioning
        # 논문에서는 Global Pooling 등으로 전체 맥락을 FiLM에 줌
        audio_ctx = memory.mean(dim=1)
        cond_emb = time_emb + audio_ctx
        
        # 3. Main Network
        h = self.proj_in(x_t)
        h = h + time_emb.unsqueeze(1)
        
        for layer in self.layers:
            h = layer(h, memory, cond_emb)
            
        return self.head(h)

class AnnealedPseudoHuberLoss(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def get_c(self, progress):
        # Linear schedule: c_max -> c_min
        alpha = progress # 0 to 1
        return (1 - alpha) * self.config.C_MAX + alpha * self.config.C_MIN

    def forward(self, audio_mert, spec, target_score, progress):
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        # 1. Get Conditions
        mert_feats = self.model.extract_mert(audio_mert)
        
        # 2. Flow Matching Setup (Optimal Transport)
        t = torch.rand(batch_size, device=device)
        x_1 = target_score
        x_0 = torch.randn_like(x_1)
        
        t_view = t.view(batch_size, 1, 1)
        # Straight path interpolation
        x_t = t_view * x_1 + (1 - t_view) * x_0
        
        # 3. Predict Velocity (Vector Field)
        pred_v = self.model(x_t, t, mert_feats, spec)
        target_v = x_1 - x_0
        
        # 4. Annealed Pseudo-Huber Loss
        diff = pred_v - target_v
        c = self.get_c(progress)
        
        loss = torch.sqrt(diff.pow(2) + c**2) - c
        return loss.mean()

    @torch.no_grad()
    def sample(self, audio_mert, spec, steps=10):
        # Flow Matching Inference (Euler ODE Solver)
        # Refinement가 필요하다면 추후 여기서 steps를 늘리거나
        # ODE Solver 대신 SDE Solver로 교체 가능하지만,
        # Flow Matching은 ODE만으로도 충분히 좋은 성능을 냅니다.
        self.model.eval()
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        mert_feats = self.model.extract_mert(audio_mert)
        seq_len = spec.size(1) 
        
        # Start from Noise (x_0)
        x_t = torch.randn(batch_size, seq_len, self.model.input_dim, device=device)
        
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((batch_size,), t_val, device=device)
            v_pred = self.model(x_t, t_tensor, mert_feats, spec)
            x_t = x_t + v_pred * dt
            
        return x_t