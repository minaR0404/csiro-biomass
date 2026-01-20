# DINOv3 No TTA PostProcess ノートブック分析

**ソース**: https://www.kaggle.com/code/mayukh18/dinov3-no-tta-postprocess
**著者**: mayukh18

---

## 1. 概要

DINOv3（Vision Transformer）をバックボーンとして使用し、Mamba風の融合ブロックを追加した画像回帰モデル。

## 2. モデルアーキテクチャ

### 2.1 バックボーン

```python
BACKBONE = "vit_huge_plus_patch16_dinov3.lvd1689m"
```

- **DINOv3 ViT-Huge+**: 最新の自己教師あり学習モデル
- timm 1.0.22を使用

### 2.2 画像処理（重要）

```python
# 画像を左右に分割
# 元画像: 2000x1000 → 左: 1000x1000, 右: 1000x1000
left = img.crop((0, 0, h, h))        # (0, 0, 1000, 1000)
right = img.crop((w - h, 0, w, h))   # (1000, 0, 2000, 1000)
```

**ポイント**: 横長画像（2:1）を正方形2枚に分割して処理

### 2.3 LocalMambaBlock（カスタム融合層）

```python
class LocalMambaBlock(nn.Module):
    """Lightweight Mamba-like block"""
    - LayerNorm
    - Depthwise Conv1d (kernel_size=5)
    - Gated Linear Unit
    - Projection + Dropout
    - Residual Connection
```

### 2.4 BiomassModel 全体構成

```
入力画像 (2000x1000)
    ↓
左右分割 → 左(1000x1000), 右(1000x1000)
    ↓
DINOv3 Backbone (各々)
    ↓
特徴量結合 (concat)
    ↓
LocalMambaBlock x 2 (融合)
    ↓
AdaptiveAvgPool1d → Flatten
    ↓
┌─────────────────────────────────┐
│  3つの独立したヘッド（Softplus出力）  │
│  ├─ head_green  → Dry_Green_g  │
│  ├─ head_dead   → Dry_Dead_g   │
│  └─ head_clover → Dry_Clover_g │
└─────────────────────────────────┘
    ↓
計算で導出:
  GDM_g = green + clover
  Dry_Total_g = GDM_g + dead
```

### 2.5 回帰ヘッド詳細

```python
self.head_green = nn.Sequential(
    nn.Linear(nf, nf // 2),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(nf // 2, 1),
    nn.Softplus()  # 出力を非負に保証
)
```

**Softplus**: 出力が負にならないよう保証（バイオマスは非負）

## 3. ハイパーパラメータ

| パラメータ | 値 |
|-----------|-----|
| IMG_SIZE | 512 |
| N_FOLDS | 4 |
| BATCH_SIZE | 2 |
| USE_TTA | False |

## 4. データ前処理

```python
test_tfms = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

- ImageNet標準の正規化を使用

## 5. アンサンブル戦略

### 5.1 使用モデル

| Fold | モデルファイル | 重み |
|------|---------------|------|
| 1 | biomass_model_easy_fold1_0.6228.pth | 0.75 |
| 3 | biomass_model_easy_fold3.pth | 1.25 |

- **加重平均**: スコアが良いfold 3に高い重みを付与
- Fold 0, 2は使用していない（モデルファイルなし/スコア低い？）

### 5.2 アンサンブル計算

```python
ensemble_predictions = np.mean(all_fold_predictions, axis=0)
# 実際は加重平均: (0.75 * fold1 + 1.25 * fold3) / 2
```

## 6. 後処理（PostProcess）

### 6.1 ターゲット別調整

```python
for target_idx, target_name in enumerate(CFG.TARGETS):
    prediction = ensemble_predictions[idx, target_idx]

    if target_name == "Dry_Clover_g":
        prediction = prediction * 0.8  # 20%減少

    elif target_name == "Dry_Dead_g":
        if prediction > 20:
            prediction *= 1.1  # 10%増加
        elif prediction < 10:
            prediction *= 0.9  # 10%減少
```

**ポイント**:
- `Dry_Clover_g`: 常に20%減少（過大予測の傾向あり？）
- `Dry_Dead_g`: 値に応じて調整（高いときは増、低いときは減）

## 7. TTA（Test Time Augmentation）

定義されているが、**USE_TTA = False** で無効化

```python
# 定義されているTTA変換
- Original
- Horizontal Flip
- Vertical Flip
- Rotate 90度
```

## 8. 予測結果例

| ターゲット | 予測値 |
|-----------|--------|
| Dry_Green_g | 37.04 |
| Dry_Dead_g | 35.25（後処理後） |
| Dry_Clover_g | 0.35（後処理後） |
| GDM_g | 37.48 |
| Dry_Total_g | 69.52 |

## 9. 学んだポイント

### 9.1 アーキテクチャ
- **画像分割**: 2:1画像を2つの正方形に分割 → 重複部分あり
- **3ターゲット予測**: Green, Dead, Cloverのみ直接予測
- **関係式活用**: GDM_g, Dry_Total_gは計算で導出
- **Softplus**: 非負出力を保証

### 9.2 学習戦略
- **K-Fold**: 4 Fold（ただし2モデルのみ使用）
- **加重アンサンブル**: 良いFoldに高い重み

### 9.3 後処理
- ターゲット別の経験的調整係数
- 検証データでの分析に基づく（推測）

## 10. 改善のアイデア

1. **全4 Foldの使用**: より安定したアンサンブル
2. **TTA有効化**: スコア向上の可能性
3. **後処理の最適化**: より精密な調整係数の探索
4. **異なるバックボーン**: EfficientNet, ConvNeXt等との比較
5. **画像分割戦略**: オーバーラップの調整、3分割など

---

*分析日: 2026年1月21日*
