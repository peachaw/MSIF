"""
MSIF: Multimodal Sentiment Interaction and Fusion
完整模型实现，严格按照论文 3.1-3.4 节描述。

架构概览:
  输入: 图像I, 文本T, 预提取区域特征
    ├─ TextEncoder (BERT-Base + BiGRU)  → H (隐藏状态), F_t (文本上下文)
    ├─ ImageEncoder (ResNet-18)          → F_v (图像全局特征)
    ├─ RegionProjection (Linear)         → R  (区域特征)
    ├─ CrossModalAlignment               → Align (对齐特征)
    ├─ FusionAttention                   → F_final (融合特征)
    └─ Classifier (Linear + Softmax)     → ŷ  (情感预测)

维度约定:
  - BERT hidden: 768
  - Region raw: 2048 → projected to 768 (d_align)
  - BiGRU hidden: 256 per direction → F_t: 256-dim
  - ResNet-18 output: 512 (F_v)
  - Fusion attention projection: 256 (d_attn)
  - F_final: 256 + 256 + 512 + 256 = 1280
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torchvision.models as models


# ===========================================================================
#  3.1 图文特征提取模块
# ===========================================================================

class TextEncoder(nn.Module):
    """
    文本特征提取器: BERT-Base + BiGRU
    
    输出:
      H:   (batch, n, 768)  — BERT 隐藏状态，用于对齐模块
      F_t: (batch, gru_hidden) — 文本上下文特征，用于融合模块
    """
    
    def __init__(self, bert_model="bert-base-uncased", gru_hidden=256,
                 freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.gru_hidden = gru_hidden
        self.bigru = nn.GRU(
            input_size=768,          # BERT hidden size
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      (batch, n) — BERT token ids
            attention_mask: (batch, n) — padding mask
        Returns:
            H:   (batch, n, 768) — BERT 隐藏状态
            F_t: (batch, gru_hidden) — 文本上下文特征 (mean pooled)
        """
        # BERT 编码
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = bert_output.last_hidden_state  # (batch, n, 768)
        
        # BiGRU 上下文建模
        gru_output, _ = self.bigru(H)  # (batch, n, 2 * gru_hidden)
        
        # 论文公式 (5): F_t = 1/2 * (forward + backward)
        forward_out = gru_output[:, :, :self.gru_hidden]      # (batch, n, gru_hidden)
        backward_out = gru_output[:, :, self.gru_hidden:]     # (batch, n, gru_hidden)
        F_t_seq = (forward_out + backward_out) / 2.0          # (batch, n, gru_hidden)
        
        # 均值池化得到全局文本上下文特征
        mask = attention_mask.unsqueeze(-1).float()            # (batch, n, 1)
        F_t = (F_t_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # (batch, gru_hidden)
        
        return H, F_t


class ImageEncoder(nn.Module):
    """
    图像全局特征提取器: ResNet-18
    
    论文公式 (6): F_v = ResNet18(I)
    输出: F_v: (batch, 512) — 去除最后全连接层的 512 维特征
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # 去除最后的全连接层，保留到 avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    
    def forward(self, image):
        """
        Args:
            image: (batch, 3, 224, 224)
        Returns:
            F_v: (batch, 512)
        """
        features = self.backbone(image)          # (batch, 512, 1, 1)
        F_v = features.flatten(start_dim=1)      # (batch, 512)
        return F_v


# ===========================================================================
#  3.2 图文特征对齐模块
# ===========================================================================

class CrossModalAlignment(nn.Module):
    """
    跨模态特征对齐模块：双向注意力 + 一致性约束 + 门控机制
    
    论文 3.2 节完整实现：
      1. 双线性评分 s(x,y) = x^T W_s y
      2. 双向注意力 a^{R→H}, a^{H→R}
      3. 双向一致性权重 ā_ij = a^{R→H}_ij * a^{H→R}_ij
      4. 门控融合 Align_i = G_i ⊙ r_i + (1-G_i) ⊙ h'_i
    
    支持消融:
      - use_gating=False:      替换门控为简单拼接+投影
      - use_bidirectional=False: 仅使用单向注意力
    """
    
    def __init__(self, d_region=768, d_text=768,
                 use_gating=True, use_bidirectional=True):
        super().__init__()
        self.d_region = d_region
        self.d_text = d_text
        self.use_gating = use_gating
        self.use_bidirectional = use_bidirectional
        
        # 双线性评分矩阵 W_s ∈ R^{d_region × d_text}
        self.W_s = nn.Parameter(torch.empty(d_region, d_text))
        nn.init.xavier_uniform_(self.W_s)
        
        if use_gating:
            # 门控参数: W_g, b_g
            self.gate_linear = nn.Linear(d_region + d_text, d_region)
        else:
            # 消融: 拼接后投影
            self.proj_linear = nn.Linear(d_region + d_text, d_region)
    
    def forward(self, R, H, text_mask=None):
        """
        Args:
            R:         (batch, m, d_region) — 投影后的区域特征
            H:         (batch, n, d_text)   — BERT 隐藏状态
            text_mask: (batch, n)           — 文本 padding mask
        Returns:
            Align:     (batch, m, d_region) — 对齐特征
        """
        batch_size, m, _ = R.shape
        _, n, _ = H.shape
        
        # ---- 1. 计算双线性评分 s(r_i, h_j) = r_i^T W_s h_j ----
        R_proj = torch.matmul(R, self.W_s)           # (batch, m, d_text)
        scores = torch.bmm(R_proj, H.transpose(1, 2))  # (batch, m, n)
        
        # 对 padding 位置 mask
        if text_mask is not None:
            pad_mask = text_mask.unsqueeze(1).expand(-1, m, -1)  # (batch, m, n)
            scores = scores.masked_fill(pad_mask == 0, -1e9)
        
        # ---- 2. 双向注意力 ----
        # R→H: 对每个区域 r_i，在文本词上分配注意力
        a_R2H = F.softmax(scores, dim=2)              # (batch, m, n)
        
        # H→R: 对每个文本词 h_j，在区域上分配注意力
        if text_mask is not None:
            scores_t = scores.transpose(1, 2)          # (batch, n, m)
        else:
            scores_t = scores.transpose(1, 2)
        a_H2R = F.softmax(scores_t, dim=2)            # (batch, n, m)
        
        # ---- 3. 计算对齐的文本表示 ----
        if self.use_bidirectional:
            # 双向一致性权重: ā_ij = a^{R→H}_ij * a^{H→R}_ji
            # a_H2R[:, j, i] 表示词 j 对区域 i 的注意力
            # 转置得到 (batch, m, n)，使得 [i, j] 位置是词 j 对区域 i 的权重
            a_H2R_t = a_H2R.transpose(1, 2)           # (batch, m, n)
            a_bar = a_R2H * a_H2R_t                   # (batch, m, n) 逐元素乘
            
            # 归一化
            a_bar_sum = a_bar.sum(dim=2, keepdim=True).clamp(min=1e-8)
            a_bar_norm = a_bar / a_bar_sum             # (batch, m, n)
            
            # h'_i = Σ_j (ā_ij_norm * h_j)
            h_prime = torch.bmm(a_bar_norm, H)         # (batch, m, d_text)
        else:
            # 消融: 仅使用单向 R→H 注意力
            # h̃_i = Σ_j (a^{R→H}_ij * h_j)
            h_prime = torch.bmm(a_R2H, H)             # (batch, m, d_text)
        
        # ---- 4. 门控机制 ----
        if self.use_gating:
            # G_i = σ(W_g [r_i; h'_i] + b_g)
            concat = torch.cat([R, h_prime], dim=-1)   # (batch, m, d_region + d_text)
            G = torch.sigmoid(self.gate_linear(concat))  # (batch, m, d_region)
            # Align_i = G_i ⊙ r_i + (1 - G_i) ⊙ h'_i
            Align = G * R + (1 - G) * h_prime          # (batch, m, d_region)
        else:
            # 消融: 简单拼接 + 线性投影
            concat = torch.cat([R, h_prime], dim=-1)
            Align = self.proj_linear(concat)           # (batch, m, d_region)
        
        return Align


# ===========================================================================
#  3.3 融合注意力模块
# ===========================================================================

class FusionAttention(nn.Module):
    """
    融合注意力模块：对齐特征 + 全局视觉上下文 + 文本上下文 的联合推理
    
    论文 3.3 节实现：
      方向1: Q=F_t, K/V=Align → V_enh (文本引导的视觉增强)
      方向2: Q=F_v, K/V=Align → T_enh (视觉引导的文本增强)
      输出:  F_final = Concat(V_enh, T_enh, F_v, F_t)
    """
    
    def __init__(self, d_align=768, d_text_ctx=256, d_visual=512, d_attn=256):
        super().__init__()
        self.d_attn = d_attn
        
        # 方向1: 文本查询 → 对齐特征的键值
        self.W_qt = nn.Linear(d_text_ctx, d_attn)    # Q from F_t
        self.W_kr = nn.Linear(d_align, d_attn)        # K from Align
        self.W_vr = nn.Linear(d_align, d_attn)        # V from Align
        
        # 方向2: 视觉查询 → 对齐特征的键值
        self.W_qv = nn.Linear(d_visual, d_attn)       # Q from F_v
        self.W_kt = nn.Linear(d_align, d_attn)        # K from Align
        self.W_vt = nn.Linear(d_align, d_attn)        # V from Align
    
    def forward(self, Align, F_t, F_v):
        """
        Args:
            Align: (batch, m, d_align)    — 对齐特征
            F_t:   (batch, d_text_ctx)    — 文本上下文特征
            F_v:   (batch, d_visual)      — 图像全局特征
        Returns:
            F_final: (batch, 2*d_attn + d_visual + d_text_ctx)
        """
        # ---- 方向1: 文本引导视觉增强 ----
        # 论文公式 Att_v = softmax(Q_t K_r^T / √d_k)
        Q_t = self.W_qt(F_t).unsqueeze(1)             # (batch, 1, d_attn)
        K_r = self.W_kr(Align)                         # (batch, m, d_attn)
        V_r = self.W_vr(Align)                         # (batch, m, d_attn)
        
        attn_scores_v = torch.bmm(Q_t, K_r.transpose(1, 2))  # (batch, 1, m)
        Att_v = F.softmax(attn_scores_v / (self.d_attn ** 0.5), dim=-1)
        V_enh = torch.bmm(Att_v, V_r).squeeze(1)      # (batch, d_attn)
        
        # ---- 方向2: 视觉引导文本增强 ----
        # 论文公式 Att_t = softmax(Q_v K_t^T / √d_k)
        Q_v = self.W_qv(F_v).unsqueeze(1)             # (batch, 1, d_attn)
        K_t = self.W_kt(Align)                         # (batch, m, d_attn)
        V_t = self.W_vt(Align)                         # (batch, m, d_attn)
        
        attn_scores_t = torch.bmm(Q_v, K_t.transpose(1, 2))  # (batch, 1, m)
        Att_t = F.softmax(attn_scores_t / (self.d_attn ** 0.5), dim=-1)
        T_enh = torch.bmm(Att_t, V_t).squeeze(1)      # (batch, d_attn)
        
        # ---- 融合 ----
        # 论文公式: F_final = Concat(V_enh, T_enh, F_v, F_t)
        F_final = torch.cat([V_enh, T_enh, F_v, F_t], dim=-1)
        
        return F_final


# ===========================================================================
#  完整 MSIF 模型
# ===========================================================================

class MSIF(nn.Module):
    """
    MSIF 完整模型
    
    支持的消融变体 (通过 config 控制):
      - use_alignment=False:     移除对齐模块 (w/o Align)
      - use_fusion=False:        移除融合模块 (w/o Fusion)
      - use_gating=False:        移除门控机制 (w/o Gating)
      - use_bidirectional=False: 移除双向一致性 (w/o Bidirectional Consistency)
      - use_bigru=False:         移除 BiGRU (w/o BiGRU)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ---------- 文本编码器 ----------
        self.bert = BertModel.from_pretrained(config.bert_model)
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        if config.use_bigru:
            self.bigru = nn.GRU(
                input_size=768,
                hidden_size=config.gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            d_text_ctx = config.gru_hidden
        else:
            d_text_ctx = 768  # 直接使用 BERT 输出
        
        # ---------- 图像编码器 ----------
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        d_visual = 512
        
        # ---------- 区域特征投影 ----------
        # 论文公式 (1): r_i = W_r f_i + b_r
        self.region_proj = nn.Linear(config.region_feat_dim, config.d_align)
        
        # ---------- 对齐模块 ----------
        if config.use_alignment:
            self.alignment = CrossModalAlignment(
                d_region=config.d_align,
                d_text=768,  # BERT hidden dim
                use_gating=config.use_gating,
                use_bidirectional=config.use_bidirectional,
            )
        
        # ---------- 融合模块 ----------
        if config.use_fusion:
            self.fusion = FusionAttention(
                d_align=config.d_align,
                d_text_ctx=d_text_ctx,
                d_visual=d_visual,
                d_attn=config.d_attn,
            )
            d_final = 2 * config.d_attn + d_visual + d_text_ctx
        else:
            # 无融合模块: 对齐特征均值池化后与 F_v, F_t 拼接
            d_final = config.d_align + d_visual + d_text_ctx
        
        # ---------- 分类器 ----------
        # 论文公式: ŷ = softmax(W_c F_final + b_c)
        self.classifier = nn.Linear(d_final, config.num_classes)
    
    def _encode_text(self, input_ids, attention_mask):
        """文本编码: BERT + 可选 BiGRU"""
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = bert_output.last_hidden_state  # (batch, n, 768)
        
        mask_float = attention_mask.unsqueeze(-1).float()  # (batch, n, 1)
        
        if self.config.use_bigru:
            gru_output, _ = self.bigru(H)  # (batch, n, 2 * gru_hidden)
            # F_t = 1/2 * (forward + backward), then mean pool
            fwd = gru_output[:, :, :self.config.gru_hidden]
            bwd = gru_output[:, :, self.config.gru_hidden:]
            F_t_seq = (fwd + bwd) / 2.0   # (batch, n, gru_hidden)
            F_t = (F_t_seq * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-8)
        else:
            # 消融: 直接使用 BERT 输出的均值池化
            F_t = (H * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-8)
        
        return H, F_t
    
    def _encode_image(self, image):
        """图像全局特征: ResNet-18"""
        features = self.image_encoder(image)  # (batch, 512, 1, 1)
        F_v = features.flatten(start_dim=1)   # (batch, 512)
        return F_v
    
    def forward(self, input_ids, attention_mask, image, region_features):
        """
        Args:
            input_ids:       (batch, n)        — BERT token ids
            attention_mask:  (batch, n)        — padding mask
            image:           (batch, 3, 224, 224) — 输入图像
            region_features: (batch, m, 2048)  — 预提取的区域特征
        Returns:
            logits: (batch, num_classes)
        """
        # 1. 文本编码
        H, F_t = self._encode_text(input_ids, attention_mask)
        
        # 2. 图像全局编码
        F_v = self._encode_image(image)
        
        # 3. 区域特征投影
        R = self.region_proj(region_features)  # (batch, m, d_align)
        
        # 4. 跨模态对齐
        if self.config.use_alignment:
            Align = self.alignment(R, H, text_mask=attention_mask)
        else:
            # 消融: 不做对齐，直接使用投影后的区域特征
            Align = R
        
        # 5. 特征融合
        if self.config.use_fusion:
            F_final = self.fusion(Align, F_t, F_v)
        else:
            # 消融: 无融合模块，对齐特征均值池化后拼接
            Align_avg = Align.mean(dim=1)  # (batch, d_align)
            F_final = torch.cat([Align_avg, F_v, F_t], dim=-1)
        
        # 6. 分类
        logits = self.classifier(F_final)
        
        return logits
    
    def count_parameters(self):
        """统计可训练参数量"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total
