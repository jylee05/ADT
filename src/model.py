# src/model.py
import torch
import torch.nn as nn
import math
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

class FlowMatchingTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.DRUM_CHANNELS * config.FEATURE_DIM
        
        # 1. Input Projection (Onset+Vel -> Hidden)
        self.proj_in = nn.Linear(self.input_dim, config.HIDDEN_DIM)
        
        # 2. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.HIDDEN_DIM),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        # 3. MERT Projection
        self.cond_proj = nn.Linear(config.MERT_DIM, config.HIDDEN_DIM)
        
        # 4. Transformer Decoder (Self-Attn + Cross-Attn)
        layer = nn.TransformerDecoderLayer(
            d_model=config.HIDDEN_DIM, 
            nhead=config.N_HEADS, 
            dim_feedforward=config.HIDDEN_DIM * 4,
            batch_first=True,
            norm_first=True # Pre-Norm for stability
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=config.N_LAYERS)
        
        # 5. Output Head (Predict Vector Field v)
        self.head = nn.Linear(config.HIDDEN_DIM, self.input_dim)
        
        # MERT Model (Frozen)
        self.mert = Wav2Vec2Model.from_pretrained("m-a-p/MERT-v1-330M", cache_dir=config.CACHE_DIR)
        self.mert.eval()
        for p in self.mert.parameters():
            p.requires_grad = False

    def extract_mert(self, audio):
        # audio: (B, T_audio) -> 24kHz
        with torch.no_grad():
            outputs = self.mert(audio, output_hidden_states=True)
        # Layer 10 is recommended by N2N authors or just last layer
        return outputs.last_hidden_state

    def forward(self, x_t, t, mert_feats):
        # x_t: (B, Seq, In_Dim), t: (B,), mert_feats: (B, M_Seq, M_Dim)
        
        # Embedding
        h = self.proj_in(x_t)
        
        # Time Inject
        time_emb = self.time_mlp(t) # (B, Hidden)
        h = h + time_emb.unsqueeze(1)
        
        # Condition Inject
        memory = self.cond_proj(mert_feats)
        
        # Transformer
        out = self.transformer(tgt=h, memory=memory)
        
        return self.head(out)

class RectifiedFlowLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, target_score):
        """
        audio: (B, Samples)
        target_score: (B, Seq_Len, 14) -> x_1
        """
        device = audio.device
        batch_size = audio.size(0)
        
        # 1. Get Conditions
        mert_feats = self.model.extract_mert(audio)
        
        # 2. Sample time t (0~1)
        t = torch.rand(batch_size, device=device)
        
        # 3. Create Noise (x_0)
        x_1 = target_score
        x_0 = torch.randn_like(x_1)
        
        # 4. Interpolate (Straight Line)
        t_view = t.view(batch_size, 1, 1)
        x_t = t_view * x_1 + (1 - t_view) * x_0
        
        # 5. Predict Velocity
        pred_v = self.model(x_t, t, mert_feats)
        target_v = x_1 - x_0
        
        # 6. Loss (MSE)
        return torch.mean((pred_v - target_v) ** 2)

    @torch.no_grad()
    def sample(self, audio, steps=10):
        self.model.eval()
        device = audio.device
        batch_size = audio.size(0)
        
        mert_feats = self.model.extract_mert(audio)
        
        # Target Seq Length (오디오 길이에 비례하여 계산)
        # MERT 24k -> output approx 75Hz (320 hop) but let's fix to user defined FPS (100Hz)
        duration_sec = audio.size(1) / 24000
        seq_len = int(duration_sec * 100) # 100 FPS
        
        # Init Noise x_0
        x_t = torch.randn(batch_size, seq_len, self.model.input_dim, device=device)
        
        # Euler Integration
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((batch_size,), t_val, device=device)
            v_pred = self.model(x_t, t_tensor, mert_feats)
            x_t = x_t + v_pred * dt
            
        return x_t