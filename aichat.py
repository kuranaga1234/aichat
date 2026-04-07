import os
import argparse
import base64
from llama_cpp import Llama

# --- 設定定数 ---
USE_LOCAL_MODEL = True
MODEL_DIR = "./models"
IMAGE_OUTPUT_DIR = "./generated_images"

MODELS = {
    # 大型モデル
    "qwen": {"name": "Qwen2.5-7B", "repo_id": "Bartowski/Qwen2.5-7B-Instruct-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 8192},
    "gemma": {"name": "Gemma-2-9B", "repo_id": "Bartowski/gemma-2-9b-it-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 4096},
    "elyza": {"name": "Llama-3-ELYZA-JP-8B", "repo_id": "elyza/Llama-3-ELYZA-JP-8B-GGUF", "filename": "Llama-3-ELYZA-JP-8B-q4_k_m.gguf", "n_ctx": 4096},
    # 軽量モデル
    "llama3b": {"name": "Llama-3.2-3B", "repo_id": "Bartowski/Llama-3.2-3B-Instruct-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 16384},
    "llama1b": {"name": "Llama-3.2-1B", "repo_id": "Bartowski/Llama-3.2-1B-Instruct-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 32768}
}

# Stable Diffusion 設定
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_IMAGE_STEPS = 20
SD_IMAGE_WIDTH = 512
SD_IMAGE_HEIGHT = 512
LORA_DIR = "./models/lora"  # LoRAファイルの保存ディレクトリ

# コマンドプレフィックス
IMAGE_GEN_PREFIX = "/image"      # 画像生成


# ─────────────────────────────────────────────
# Stable Diffusion（画像生成）
# ─────────────────────────────────────────────
def load_sd_pipeline():
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        from pathlib import Path

        print(f"--- Stable Diffusion ロード中: {SD_MODEL_ID} (CPU モード) ---")
        print("    ※ 初回はモデルのダウンロードが発生します（約2〜4GB）")

        # ローカルキャッシュディレクトリを設定
        sd_cache_dir = os.path.join(MODEL_DIR, "stable-diffusion")
        os.makedirs(sd_cache_dir, exist_ok=True)

        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=sd_cache_dir
        )
               
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        
        # LoRA ウェイトを読み込む
        if os.path.exists(LORA_DIR):
            lora_files = list(Path(LORA_DIR).glob("*.safetensors")) + list(Path(LORA_DIR).glob("*.pt"))
            if lora_files:
                lora_path = str(lora_files[0])  # 最初のLoRAファイルを使用
                try:
                    print(f"[LoRA] ロード中: {os.path.basename(lora_path)}")
                    pipe.load_lora_weights(lora_path)
                    print(f"[LoRA] ロード完了: {os.path.basename(lora_path)}")
                except Exception as e:
                    print(f"[LoRA] ロード失敗: {e}")
            else:
                print(f"[LoRA] {LORA_DIR} にLoRAファイルが見つかりません。")
        else:
            print(f"[LoRA] ディレクトリが見つかりません: {LORA_DIR}")
        
        print("--- Stable Diffusion ロード完了 ---")
        return pipe

    except ImportError:
        print("\n[エラー] diffusers または torch がインストールされていません。")
        print("  pip install diffusers transformers torch accelerate")
        return None


def generate_image(pipe, prompt: str, save_dir: str):
    if pipe is None:
        print("[画像生成] Stable Diffusion が初期化されていません。")
        return

    import torch
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"image_{timestamp}.png")

    print(f"\n[画像生成] プロンプト: {prompt}")
    print(f"[画像生成] 生成中... （CPU では数分かかります）")

    try:
        with torch.no_grad():
            result = pipe(prompt=prompt, num_inference_steps=SD_IMAGE_STEPS,
                          width=SD_IMAGE_WIDTH, height=SD_IMAGE_HEIGHT)
        result.images[0].save(filepath)
        print(f"[画像生成] 完了！保存先: {filepath}")
    except Exception as e:
        print(f"[画像生成] エラー: {e}")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Llama-cpp AI Chat with Image Generation & Analysis")
    parser.add_argument("--model", type=str, default="qwen", choices=list(MODELS.keys()))
    parser.add_argument("--file", type=str, help="読み込むファイルのパスを指定してください")
    parser.add_argument("--enable-image", action="store_true",
                        help="Stable Diffusion による画像生成を有効にする")
    args = parser.parse_args()

    # ファイル読み込み
    file_content = ""
    if args.file:
        if os.path.exists(args.file):
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    file_content = f.read()
                print(f"--- ファイル '{args.file}' を読み込みました ---")
            except Exception as e:
                print(f"ファイル読み込みエラー: {e}")
        else:
            print(f"警告: ファイル '{args.file}' が見つかりません。")

    # Stable Diffusion ロード
    sd_pipe = None
    if args.enable_image:
        sd_pipe = load_sd_pipeline()

    # テキストモデルロード
    selected = MODELS[args.model]
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"--- ロード中: {selected['name']} (n_ctx: {selected['n_ctx']}) ---")
    llm = Llama.from_pretrained(
        repo_id=selected["repo_id"],
        filename=selected["filename"],
        local_dir=MODEL_DIR,
        offline=USE_LOCAL_MODEL,
        verbose=False,
        n_ctx=selected["n_ctx"],
        n_batch=512
    )

    history = []
    print(f"\n--- {selected['name']} とのチャット開始 ---")
    if args.enable_image:
        print(f"  {IMAGE_GEN_PREFIX} <英語プロンプト>        → 画像を生成")
    else:
        print(f"  画像生成: 無効（--enable-image で有効化）")

    while True:
        user_input = input("\nあなた: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', '終了']:
            break

        # ===== 画像生成コマンド =====
        if user_input.startswith(IMAGE_GEN_PREFIX):
            if not args.enable_image:
                print(f"[画像生成] 無効です。--enable-image を付けて起動してください。")
                continue
            prompt = user_input[len(IMAGE_GEN_PREFIX):].strip()
            if not prompt:
                print(f"例: {IMAGE_GEN_PREFIX} a beautiful mountain landscape at sunset")
                continue
            generate_image(sd_pipe, prompt, IMAGE_OUTPUT_DIR)
            continue

        # ===== 通常チャット =====
        prompt = user_input
        if len(history) == 0:
            instruction = "あなたは優秀なアシスタントです。提供されたファイルの内容を元に回答してください。\n\n"
            if file_content:
                prompt = f"{instruction}[ファイルの内容]:\n{file_content}\n\n[ユーザーの質問]: {user_input}"
            else:
                prompt = f"{instruction}[ユーザーの質問]: {user_input}"

        history.append({"role": "user", "content": prompt})

        try:
            response = llm.create_chat_completion(messages=history, max_tokens=512)
            ai_answer = response["choices"][0]["message"]["content"]
            print(f"\nAI ({selected['name']}): {ai_answer}")
            history.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            break


if __name__ == "__main__":
    main()
