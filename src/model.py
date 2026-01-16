# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from transformers import Wav2Vec2Model

# -------------------------------------------------------------------
# Positional Embedding (Time Step용) - Diffusion/Flow Matching의 t 임베딩
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
        
        # [추가] 극단값 방지
        x_clamped = torch.clamp(x, min=-10.0, max=10.0)  # 시간 스텍 극단값 제한
        
        emb = x_clamped[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# -------------------------------------------------------------------
# Sequence Positional Embedding (시퀀스 위치 정보용)
# -------------------------------------------------------------------
class SequencePositionalEncoding(nn.Module):
    """Transformer 시퀀스용 Sinusoidal Positional Encoding"""
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (Batch, Seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

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
        
        # [수정] Scale 값을 제한하여 gradient explosion 방지
        scale = torch.tanh(scale) * 0.5  # [-0.5, 0.5] 범위로 제한
        
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
        
        # 2. Time Embedding MLP (Flow Matching의 t용)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.HIDDEN_DIM),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        # [추가] 3. Sequence Positional Encodings
        # 각 시퀀스의 위치 정보를 인코딩
        self.pos_enc_main = SequencePositionalEncoding(config.HIDDEN_DIM, max_len=2000)
        self.pos_enc_mert = SequencePositionalEncoding(config.HIDDEN_DIM, max_len=1000)
        self.pos_enc_spec = SequencePositionalEncoding(config.HIDDEN_DIM, max_len=1000)
        
        # 4. Condition Encoders (분리형 구조)
        # 4-1. MERT Path
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

        # 4-2. Spectrogram Path
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
        
        # 5. Learned Null Embeddings (학습 가능한 0 벡터)
        # Dropout 시 0 대신 이 벡터를 사용함
        self.null_mert_emb = nn.Parameter(torch.randn(1, 1, config.MERT_DIM) * 0.02)
        self.null_spec_emb = nn.Parameter(torch.randn(1, 1, config.N_MELS) * 0.02)
        
        # 6. Main Decoder Layers
        self.layers = nn.ModuleList([
            FiLMTransformerDecoderLayer(
                d_model=config.HIDDEN_DIM, 
                nhead=config.N_HEADS, 
                dim_feedforward=config.HIDDEN_DIM * 4
            )
            for _ in range(config.N_LAYERS)
        ])
        
        # 7. Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, self.input_dim)
        
        # 8. Pretrained MERT Model Load
        print(f"Loading MERT from {config.MERT_PATH}...")
        self.mert = Wav2Vec2Model.from_pretrained(config.MERT_PATH, cache_dir=config.CACHE_DIR)
        self.mert.eval()
        for p in self.mert.parameters():
            p.requires_grad = False
        
        # 9. Weight Initialization (gradient explosion 방지)
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming 초기화로 gradient explosion 방지"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform 초기화 (tanh/sigmoid용)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm 표준 초기화
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Null embeddings는 더 작게 초기화
        nn.init.normal_(self.null_mert_emb, 0, 0.01)
        nn.init.normal_(self.null_spec_emb, 0, 0.01)
    
    def extract_mert(self, audio):
        """MERT 모델에서 특정 레이어 특징 추출"""
        with torch.no_grad():
            # [추가] 입력 오디오 범위 체크
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                print("[WARNING] NaN/Inf in audio input to MERT")
                audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            outputs = self.mert(audio, output_hidden_states=True)
            features = outputs.hidden_states[self.config.MERT_LAYER_IDX]
            
            # [추가] MERT 출력 체크
            if torch.isnan(features).any():
                print("[WARNING] NaN in MERT features")
                features = torch.nan_to_num(features, nan=0.0)
                
            return features

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

    def forward(self, x_t, t, audio_mert, spec_feats):
        """
        Args:
            x_t: Noisy drum grid (B, T, D)
            t: Time step (B,)
            audio_mert: Raw audio for MERT (B, samples) - MERT 추출을 내부에서 수행!
            spec_feats: Mel-spectrogram features (B, T, N_MELS)
        """
        # 0. MERT Feature Extraction (forward 내부에서 수행하여 DataParallel 병렬화!)
        # 이렇게 해야 DataParallel이 audio_mert를 GPU별로 분할하여 병렬 처리함
        mert_feats = self.extract_mert(audio_mert)
        
        # 1. Condition Dropout & Substitution
        mert_h = self.apply_condition_dropout(
            mert_feats, self.null_mert_emb, 
            self.config.DROP_MERT_PROB, self.config.DROP_PARTIAL_PROB
        )
        spec_h = self.apply_condition_dropout(
            spec_feats, self.null_spec_emb,
            self.config.DROP_SPEC_PROB, self.config.DROP_PARTIAL_PROB
        )
        
        # 2. MERT를 Spec 길이에 맞게 Interpolate (Encoder 전에!)
        # 시간 정렬을 먼저 해야 positional encoding과 encoder가 
        # 동일한 시간 해상도에서 동작함
        target_len = spec_h.shape[1]  # Spec의 시간 길이 기준
        if mert_h.shape[1] != target_len:
            # (B, T, D) -> (B, D, T) -> interpolate -> (B, D, T') -> (B, T', D)
            mert_h = mert_h.permute(0, 2, 1)
            mert_h = F.interpolate(mert_h, size=target_len, mode='linear', align_corners=False)
            mert_h = mert_h.permute(0, 2, 1)
        
        # 3. Project to hidden dim
        mert_emb = self.mert_proj(mert_h)
        spec_emb = self.spec_proj(spec_h)
        
        # 4. Positional Encoding (시간 정렬 후 적용!)
        mert_emb = self.pos_enc_mert(mert_emb)
        spec_emb = self.pos_enc_spec(spec_emb)
        
        # 5. Encode Separately
        mert_emb = self.mert_encoder(mert_emb)
        spec_emb = self.spec_encoder(spec_emb)
        
        # 6. Concatenate (Cross Attention Memory용)
        # 이제 mert_emb와 spec_emb가 같은 시간 길이를 가짐
        memory = torch.cat([mert_emb, spec_emb], dim=1)
        
        # Global Condition for FiLM에서 NaN 방지
        time_emb = self.time_mlp(t)
        
        # [추가] Time embedding NaN 체크
        if torch.isnan(time_emb).any():
            print("[WARNING] NaN in time_emb")
            time_emb = torch.nan_to_num(time_emb, nan=0.0)
        
        # 오디오 전체 맥락(Average Pooling)을 Time 정보와 더함
        audio_ctx = memory.mean(dim=1)
        
        # [추가] Audio context NaN 체크  
        if torch.isnan(audio_ctx).any():
            print("[WARNING] NaN in audio_ctx")
            audio_ctx = torch.nan_to_num(audio_ctx, nan=0.0)
            
        cond_emb = time_emb + audio_ctx
        
        # 8. Main Network Flow
        h = self.proj_in(x_t)
        
        # [추가] proj_in 출력 NaN 체크
        if torch.isnan(h).any():
            print("[WARNING] NaN in proj_in output")
            h = torch.nan_to_num(h, nan=0.0)
        
        h = self.pos_enc_main(h)  # 메인 시퀀스에도 Positional Encoding
        # Input에도 Time 정보 더해주기 (일반적 관행)
        h = h + time_emb.unsqueeze(1)
        
        for layer in self.layers:
            h = layer(h, memory, cond_emb)
            # [추가] 각 layer 후 NaN 체크
            if torch.isnan(h).any():
                print("[WARNING] NaN in transformer layer output")
                h = torch.nan_to_num(h, nan=0.0)
                break  # NaN 발생 시 조기 종료
            
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
        """Annealing: 학습 초기에는 큰 c(MSE처럼), 후반에는 작은 c(L1처럼)"""
        alpha = progress
        return (1 - alpha) * self.config.C_MAX + alpha * self.config.C_MIN
    
    def sample_time(self, batch_size, device):
        """
        Flow Matching Time Sampling
        학습 안정성을 위해 [eps, 1-eps] 범위로 클리핑
        """
        eps = 1e-4
        t = torch.rand(batch_size, device=device)
        t = t * (1 - 2 * eps) + eps  # [eps, 1-eps]
        return t

    def forward(self, audio_mert, spec, target_score, progress):
        """
        Args:
            audio_mert: Raw audio for MERT (B, samples)
            spec: Mel-spectrogram (B, T, N_MELS)
            target_score: Ground truth drum grid (B, T, D)
            progress: Training progress [0, 1]
        """
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        # Flow Matching Setup
        t = self.sample_time(batch_size, device)
        
        x_1 = target_score        
        x_0 = torch.randn_like(x_1) 
        
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        t_view = t.view(batch_size, 1, 1)
        x_t = (1 - t_view) * x_0 + t_view * x_1
        
        # [핵심 수정] Velocity Prediction
        # audio_mert를 직접 넘겨서 DataParallel이 자동으로 GPU별 분할 처리
        # MERT 추출은 model.forward() 내부에서 수행됨
        pred_v = self.model(x_t, t, audio_mert, spec)
        
        # Target velocity: v = dx/dt = x_1 - x_0 (constant for linear path)
        target_v = x_1 - x_0 
        
        # Annealed Pseudo-Huber Loss
        diff = pred_v - target_v
        c = self.get_c(progress)
        loss = torch.sqrt(diff.pow(2) + c**2) - c
        return loss.mean()

    @torch.no_grad()
    def sample(self, audio_mert, spec, steps=10, init_score=None, start_t=0.0):
        """Inference용 샘플링 (단일 GPU 가정)"""
        self.model.eval()
        device = audio_mert.device
        batch_size = audio_mert.size(0)
        
        # Inference 시에는 모델 내부에서 MERT 추출하므로 input_dim만 가져옴
        if isinstance(self.model, nn.DataParallel):
            input_dim = self.model.module.input_dim
        else:
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
            
            # audio_mert를 직접 전달
            v_pred = self.model(x_t, t_tensor, audio_mert, spec)
            x_t = x_t + v_pred * dt
            
        return x_t