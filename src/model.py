# src/model.py
import torch
import torch.nn as nn
import math
import random
from transformers import Wav2Vec2Model

# -------------------------------------------------------------------
# Positional Embedding (Time Step용)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation) Layer
# -------------------------------------------------------------------
class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, condition):
        """
        x: (Batch, Seq, Dim)
        condition: (Batch, Cond_Dim)
        """
        # Scale과 Shift 계수 예측
        params = self.proj(condition)
        scale, shift = params.chunk(2, dim=-1)
        
        # 차원 맞추기 (Broadcasting)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        # Affine 변환
        return x * (1 + scale) + shift

# -------------------------------------------------------------------
# FiLM 기반 Transformer Decoder Layer
# -------------------------------------------------------------------
class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        # Attention Layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Normalization Layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # FiLM Layers (각 서브모듈 앞/뒤에 적용)
        self.film1 = FiLMLayer(d_model, d_model)
        self.film2 = FiLMLayer(d_model, d_model)
        self.film3 = FiLMLayer(d_model, d_model)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, tgt, memory, cond_emb):
        """
        tgt: Decoder 입력 (Noisy Grid)
        memory: Cross Attention용 Context (Spec + MERT)
        cond_emb: FiLM용 Global Condition (Time + Audio Summary)
        """
        # 1. Self Attention Block
        tgt2 = self.norm1(tgt)
        tgt2 = self.film1(tgt2, cond_emb)  # FiLM 적용
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2)
        tgt = tgt + self.dropout(tgt2)
        
        # 2. Cross Attention Block
        tgt2 = self.norm2(tgt)
        tgt2 = self.film2(tgt2, cond_emb)  # FiLM 적용
        tgt2, _ = self.cross_attn(tgt2, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        
        # 3. Feed Forward Block
        tgt2 = self.norm3(tgt)
        tgt2 = self.film3(tgt2, cond_emb)  # FiLM 적용
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

# -------------------------------------------------------------------
# 메인 모델: Flow Matching Transformer
# -------------------------------------------------------------------
class FlowMatchingTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.DRUM_CHANNELS * config.FEATURE_DIM
        
        # 1. Input Projection
        self.proj_in = nn.Linear(self.input_dim, config.HIDDEN_DIM)
        
        # 2. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.HIDDEN_DIM),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        # 3. Condition Encoders (분리형 구조)
        # 3-1. MERT Path
        self.mert_proj = nn.Linear(config.MERT_DIM, config.HIDDEN_DIM)
        self.mert_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=4, 
                dim_feedforward=config.HIDDEN_DIM*2,
                batch_first=True, 
                norm_first=True
            ),
            num_layers=config.COND_LAYERS
        )

        # 3-2. Spectrogram Path
        self.spec_proj = nn.Linear(config.N_MELS, config.HIDDEN_DIM)
        self.spec_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=4, 
                dim_feedforward=config.HIDDEN_DIM*2,
                batch_first=True, 
                norm_first=True
            ),
            num_layers=config.COND_LAYERS
        )
        
        # 4. Learned Null Embeddings (학습 가능한 0 벡터)
        # Dropout 시 0 대신 이 벡터를 사용함
        self.null_mert_emb = nn.Parameter(torch.randn(1, 1, config.MERT_DIM) * 0.02)
        self.null_spec_emb = nn.Parameter(torch.randn(1, 1, config.N_MELS) * 0.02)
        
        # 5. Main Decoder Layers
        self.layers = nn.ModuleList([
            FiLMTransformerDecoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=config.N_HEADS, 
                dim_feedforward=config.HIDDEN_DIM * 4
            )
            for _ in range(config.N_LAYERS)
        ])
        
        # 6. Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, self.input_dim)
        
        # 7. Pretrained MERT Model Load
        print(f"Loading MERT from {config.MERT_PATH}...")
        self.mert = Wav2Vec2Model.from_pretrained(config.MERT_PATH, cache_dir=config.CACHE_DIR)
        self.mert.eval()
        for p in self.mert.parameters():
            p.requires_grad = False

    def extract_mert(self, audio):
        """MERT 모델에서 특정 레이어 특징 추출"""
        with torch.no_grad():
            outputs = self.mert(audio, output_hidden_states=True)
            return outputs.hidden_states[self.config.MERT_LAYER_IDX]

    def apply_condition_dropout(self, feat, null_emb, drop_prob, partial_prob):
        """
        Dropouts:
        1. Complete Dropout: 배치 내 특정 샘플의 전체 컨디션을 날림
        2. Partial Dropout: 시간축의 일부 구간을 마스킹 (Inpainting 학습용)
        * 0 대신 Learned Null Embedding 사용
        """
        if not self.training:
            return feat
        
        B, T, D = feat.shape
        device = feat.device
        out_feat = feat.clone()
        
        # 1. Complete Dropout
        drop_mask = torch.bernoulli(torch.full((B, 1, 1), drop_prob, device=device)).bool()
        # 마스킹 된 곳은 null_emb로 교체
        out_feat = torch.where(drop_mask, null_emb, out_feat)
        
        # 2. Partial Dropout (Random Length)
        if random.random() < partial_prob:
            # 10% ~ 50% 길이 랜덤 마스킹
            mask_ratio = random.uniform(0.1, 0.5)
            mask_len = int(T * mask_ratio)
            
            if mask_len > 0:
                start = random.randint(0, T - mask_len)
                out_feat[:, start:start+mask_len, :] = null_emb

        return out_feat

    def forward(self, x_t, t, mert_feats, spec_feats):
        # 1. Condition Dropout & Substitution
        mert_h = self.apply_condition_dropout(
            mert_feats, self.null_mert_emb, 
            self.config.DROP_MERT_PROB, self.config.DROP_PARTIAL_PROB
        )
        spec_h = self.apply_condition_dropout(
            spec_feats, self.null_spec_emb,
            self.config.DROP_SPEC_PROB, self.config.DROP_PARTIAL_PROB
        )
        
        # 2. Encode Separately (독립된 인코더 사용)
        mert_emb = self.mert_proj(mert_h)
        mert_emb = self.mert_encoder(mert_emb)
        
        spec_emb = self.spec_proj(spec_h)
        spec_emb = self.spec_encoder(spec_emb)
        
        # 3. Concatenate (Cross Attention Memory용)
        memory = torch.cat([mert_emb, spec_emb], dim=1)
        
        # 4. Global Condition for FiLM
        time_emb = self.time_mlp(t)
        # 오디오 전체 맥락(Average Pooling)을 Time 정보와 더함
        audio_ctx = memory.mean(dim=1)
        cond_emb = time_emb + audio_ctx
        
        # 5. Main Network Flow
        h = self.proj_in(x_t)
        # Input에도 Time 정보 더해주기 (일반적 관행)
        h = h + time_emb.unsqueeze(1)
        
        for layer in self.layers:
            h = layer(h, memory, cond_emb)
            
        return self.head(h)

# -------------------------------------------------------------------
# Loss Wrapper: Annealed Pseudo-Huber Loss
# -------------------------------------------------------------------
class AnnealedPseudoHuberLoss(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def get_c(self, progress):
        alpha = progress
        return (1 - alpha) * self.config.C_MAX + alpha * self.config.C_MIN

    def forward(self, audio_mert, spec, target_score, progress):
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        # [수정] DataParallel 사용 시 .module로 접근
        if isinstance(self.model, nn.DataParallel):
            mert_feats = self.model.module.extract_mert(audio_mert)
        else:
            mert_feats = self.model.extract_mert(audio_mert)
        
        # Flow Matching Setup
        t = torch.rand(batch_size, device=device)
        
        x_1 = target_score        
        x_0 = torch.randn_like(x_1) 
        
        t_view = t.view(batch_size, 1, 1)
        x_t = t_view * x_1 + (1 - t_view) * x_0
        
        # Velocity Prediction (Forward는 DataParallel이 알아서 분배)
        pred_v = self.model(x_t, t, mert_feats, spec)
        target_v = x_1 - x_0 
        
        # Loss
        diff = pred_v - target_v
        c = self.get_c(progress)
        loss = torch.sqrt(diff.pow(2) + c**2) - c
        return loss.mean()

    @torch.no_grad()
    def sample(self, audio_mert, spec, steps=10, init_score=None, start_t=0.0):
        self.model.eval()
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        # [수정] 여기도 동일하게 .module 체크
        if isinstance(self.model, nn.DataParallel):
            mert_feats = self.model.module.extract_mert(audio_mert)
            input_dim = self.model.module.input_dim
        else:
            mert_feats = self.model.extract_mert(audio_mert)
            input_dim = self.model.input_dim

        seq_len = spec.size(1) 
        
        if init_score is not None and start_t > 0:
            noise = torch.randn_like(init_score)
            x_t = start_t * init_score + (1 - start_t) * noise
            t_current = start_t
        else:
            x_t = torch.randn(batch_size, seq_len, input_dim, device=device)
            t_current = 0.0
            
        steps_to_run = int(steps * (1.0 - t_current))
        if steps_to_run < 1: steps_to_run = 1
        
        dt = (1.0 - t_current) / steps_to_run
        
        for i in range(steps_to_run):
            t_val = t_current + i * dt
            t_tensor = torch.full((batch_size,), t_val, device=device)
            
            v_pred = self.model(x_t, t_tensor, mert_feats, spec)
            x_t = x_t + v_pred * dt
            
        return x_t