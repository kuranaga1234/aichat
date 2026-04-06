わかりました！Stable Diffusion をCPUで動かせるよう、`diffusers` + `torch`（CPU）を使って `aichat.py` に画像生成機能を追加します。主な変更点と使い方を説明します。

**追加した機能：**

画像生成は `--enable-image` フラグで明示的に有効化する設計にしました。SDのロードは重いため、不要な時はスキップできます。

**インストール（初回のみ）：**
```bash
pip install diffusers transformers torch accelerate
```

**起動方法：**
```bash
# テキストチャットのみ（従来通り）
python aichat.py

# 画像生成も使う場合
python aichat.py --enable-image

# ファイル付き＋画像生成
python aichat.py --file dataset.jsonl --enable-image
```

**チャット中の画像生成：**
```
あなた: /image a beautiful mountain landscape at sunset
```
`./generated_images/` フォルダに PNG で保存されます。

**CPU環境での注意点：**
- 初回起動時にモデルが約2〜4GBダウンロードされます
- 1枚の生成に **3〜10分** かかることがあります（`SD_IMAGE_STEPS = 20` で調整可）
- より速くしたい場合は `SD_IMAGE_STEPS` を `10` に下げると半分程度の時間になります（品質は落ちます）
- プロンプトは**英語**で入力すると精度が上がります

#### 〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇ーーー〇

わかりました！LLaVA-1.5-7BをGGUF形式でllama-cppから使う実装を追加します。既存の構成と統一感を持たせた形で組み込みます。追加内容と使い方を説明します。

**初回セットアップ（モデルのダウンロード）：**
```bash
# LLaVA の2つのGGUFファイルをダウンロード（合計約5GB）
huggingface-cli download mys/ggml_llava-v1.5-7b \
    ggml-model-q4_k.gguf mmproj-model-f16.gguf \
    --local-dir ./models
```

**起動方法：**
```bash
# 画像分析のみ有効
python aichat.py --enable-vision

# 画像生成＋分析の両方有効
python aichat.py --enable-image --enable-vision
```

**チャット中の使い方：**
```
# 質問なし（デフォルトで「詳しく説明して」）
あなた: /analyze ./photo.jpg

# 質問付き
あなた: /analyze ./photo.jpg この画像に写っている物体は何ですか？
あなた: /analyze ./dog.png この犬の犬種を教えてください。
```

**ポイント：**
- LLaVAはllama-cppと同じGGUF形式なので、既存の構成と統一感があります
- `mmproj-model-f16.gguf` は画像をベクトルに変換する「視覚エンコーダ」で、本体GGUFと必ずセットで必要です
- 質問を省略するとデフォルトで「日本語で詳しく説明してください」が送られます
- CPUでも動きますが、1回の分析に30秒〜2分程度かかります
