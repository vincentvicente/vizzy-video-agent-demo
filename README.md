# Vizzy AI Video Agent — Prototype

数据驱动的 paid-social 视频生成 agent。粘 URL → 自动产出 9:16 vertical 30-60s 广告视频。

## 锁定的架构决策（prototype 范围）

| 维度 | 决策 | 备注 |
|---|---|---|
| HITL 节奏 | **γ** · clip gen 前 1 个合并 checkpoint + final accept | 用户只点 2 次按钮 |
| Checkpoint 透明度 | **Level 1**（prototype 限定）| **产品版必须升 Level 3 + 非对称 + 双模式** |
| 记忆 | **Layer 0**（无记忆，每次从零）| 产品版升 Layer 2（brand 级 memory） |
| Schema 紧度 | **B** · role 从 enum 选, LLM 排序 + 定 duration | 给 LLM 在不同 brand 上展现智能 |
| Reference 路径 | **A.1** · 纯用户上传 | PRD 第 3 条 rationale 退化为 "用户期望" |
| 模型选择 | **Seedance 2.0 单模型**（prototype 简化）| 产品版恢复硬约束 + 偏好 + LLM 微调三层 |
| Retry 模式 | **模式 3 · 条件图路由表** | QA fault_type → 显式 retry from stage |

## Pipeline 架构

```
URL
 ↓
[Strategist]  fetch URL → brand JSON + storyboard JSON (strict schema)
 ↓
[Reference Selector]  用户上传 → 标准化存储（A.1 路线无 LLM 调用）
 ↓
[Director]  storyboard → Seedance API params (per scene)
 ↓
 ↓  ✋ γ Checkpoint：用户 approve 才花钱跑下一步
 ↓
[Clip Generation]  并行 fal.ai → Seedance 2.0
 ↓
[Editor]  ffmpeg 拼接 + ElevenLabs VO + 字幕 overlay
 ↓
[QA]  Claude vision 抽帧 → fault_type
 ↓
  ├─ pass → final video
  └─ fail → 条件图路由表 → retry from <stage>
 ↓
✋ Final accept / regen
```

## Tech Stack

- **LLM**: Anthropic Claude (Strategist / Director / QA vision)
- **视频生成**: fal.ai → Seedance 2.0 Pro (image-to-video，支持 reference 控制)
- **语音 VO**: ElevenLabs
- **拼接**: ffmpeg-python
- **UI**: Streamlit

## 跑起来

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 API keys
streamlit run app.py
```

## 设计原则

- **legibility-first**: 每个 stage I/O 都是 strict JSON, 失败时可下钻
- **不可控/untraceable 不能容忍**: 硬路由表 + 写死的 retry budget = LLM 不能跑偏
- **No visible text in model**: 所有字幕 / logo / CTA 文字在 Editor 层加 overlay, 不让视频模型生成 (规避模型乱出英文/拼写错)

## 局限 (prototype scope)

- **每次跑 3-6 分钟**: clip gen 物理瓶颈
- **每次 cost $5-15**: 主要是 fal.ai (5-6 scene × ~$1)
- **demo 友好的 fallback**: data/final/fallback.mp4 是预录的完整 run, 现场 API 失败时手动放
