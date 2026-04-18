import os
import argparse
import json
from llama_cpp import Llama
import torch
from PIL import Image

# --- 設定定数 ---
USE_LOCAL_MODEL = True
MODEL_DIR = "./models"
IMAGE_OUTPUT_DIR = "./generated_images"

MODELS = {
    # 大型モデル
    "qwen": {"name": "Qwen2.5-7B", "repo_id": "Bartowski/Qwen2.5-7B-Instruct-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 4096, "type": "llama_cpp"},
    "gemma": {"name": "Gemma-2-9B", "repo_id": "Bartowski/gemma-2-9b-it-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 4096, "type": "llama_cpp"},
    "gemma4-multi": {"name": "Gemma-4-E4B", "repo_id": "google/gemma-4-e4b-it", "n_ctx": 4096, "type": "transformers", "multimodal": True},
    "gemma4-text": {"name": "Gemma-4-E4B-GGUF", "filename": "google_gemma-4-E4B-it-Q4_K_M.gguf", "n_ctx": 4096, "type": "llama_cpp"},
    "elyza": {"name": "Llama-3-ELYZA-JP-8B", "repo_id": "elyza/Llama-3-ELYZA-JP-8B-GGUF", "filename": "Llama-3-ELYZA-JP-8B-q4_k_m.gguf", "n_ctx": 4096, "type": "llama_cpp"},
    # 軽量モデル
    "llama3b": {"name": "Llama-3.2-3B", "repo_id": "Bartowski/Llama-3.2-3B-Instruct-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 16384, "type": "llama_cpp"},
    "llama1b": {"name": "Llama-3.2-1B", "repo_id": "Bartowski/Llama-3.2-1B-Instruct-GGUF", "filename": "*Q4_K_M.gguf", "n_ctx": 32768, "type": "llama_cpp"}
}

# Stable Diffusion 設定
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_MODEL_PATH = "./models/chilloutmix_NiPrunedFp32Fix.safetensors"  # うまく使いこなせない★
SD_IMAGE_STEPS = 20
SD_IMAGE_WIDTH = 512
SD_IMAGE_HEIGHT = 512
LORA_DIR = "./models/lora"  # LoRAファイルの保存ディレクトリ

# FLUX 設定(マシンの性能が追い付いていないので使えない★)
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
FLUX_CACHE_DIR = os.path.join(MODEL_DIR, "flux-schnell")
FLUX_IMAGE_STEPS = 4       # schnell は少ステップで高品質
FLUX_IMAGE_WIDTH = 512
FLUX_IMAGE_HEIGHT = 512

# コマンドプレフィックス
IMAGE_GEN_PREFIX = "/image"      # 画像生成
FLUX_GEN_PREFIX  = "/flux"       # FLUX画像生成
ANALYZE_PREFIX = "/analyze"      # 画像分析（Gemma-4用）

# ─────────────────────────────────────────────
# Stable Diffusion（画像生成）
# ─────────────────────────────────────────────
def load_sd_pipeline(is_default):
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        from pathlib import Path

        if is_default:
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
        else:
            print(f"--- Stable Diffusion ロード中: {SD_MODEL_PATH} ---")

            # ローカルキャッシュディレクトリを設定
            sd_cache_dir = os.path.join(MODEL_DIR, "stable-diffusion")
            os.makedirs(sd_cache_dir, exist_ok=True)

            pipe = StableDiffusionPipeline.from_single_file(
                SD_MODEL_PATH,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
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


def generate_image(pipe, prompt: str, save_dir: str, negative_prompt: str = ""):
    if pipe is None:
        print("[画像生成] Stable Diffusion が初期化されていません。")
        return

    import torch
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"image_{timestamp}.png")

    print(f"\n[画像生成] プロンプト: {prompt}")
    if negative_prompt:
        print(f"[画像生成] ネガティブプロンプト: {negative_prompt}")
    print(f"[画像生成] 生成中... （CPU では数分かかります）")

    try:
        with torch.no_grad():
            kwargs = dict(
                prompt=prompt,
                num_inference_steps=SD_IMAGE_STEPS,
                width=SD_IMAGE_WIDTH,
                height=SD_IMAGE_HEIGHT,
            )
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt
            result = pipe(**kwargs)
        result.images[0].save(filepath)
        print(f"[画像生成] 完了！保存先: {filepath}")
    except Exception as e:
        print(f"[画像生成] エラー: {e}")

def load_flux_pipeline():
    try:
        from diffusers import FluxPipeline
        import torch

        os.makedirs(FLUX_CACHE_DIR, exist_ok=True)

        if torch.cuda.is_available():
            dtype = torch.bfloat16
            device = "cuda"
            print("--- FLUX.1-schnell ロード中 (GPU / bfloat16) ---")
        else:
            dtype = torch.float32
            device = "cpu"
            print("--- FLUX.1-schnell ロード中 (CPU / float32) ---")
            print("    ※ CPU での推論は非常に時間がかかります（30分以上の場合あり）")

        print("    ※ 初回はモデルのダウンロードが発生します（約 24GB）")

        # HF_TOKEN 環境変数からトークンを取得
        hf_token = os.environ.get("HF_TOKEN", None)

        pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL_ID,
            torch_dtype=dtype,
            cache_dir=FLUX_CACHE_DIR,
            token=hf_token,
        )
        pipe = pipe.to(device)

        if device == "cpu":
            pipe.enable_attention_slicing()
        elif torch.cuda.get_device_properties(0).total_memory < 16 * 1024 ** 3:
            pipe.enable_sequential_cpu_offload()

        print("--- FLUX.1-schnell ロード完了 ---")
        return pipe

    except ImportError:
        print("\n[エラー] diffusers または torch がインストールされていません。")
        print("  pip install diffusers transformers torch accelerate sentencepiece")
        return None
    except Exception as e:
        print(f"\n[FLUX ロードエラー] {e}")
        return None


def generate_flux_image(pipe, prompt: str, save_dir: str):
    """FLUX は negative_prompt 非対応"""
    if pipe is None:
        print("[FLUX] パイプラインが初期化されていません。")
        return

    import torch
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"flux_{timestamp}.png")

    print(f"\n[FLUX] プロンプト: {prompt}")
    print(f"[FLUX] 生成中... （ステップ数: {FLUX_IMAGE_STEPS}）")

    try:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                num_inference_steps=FLUX_IMAGE_STEPS,
                width=FLUX_IMAGE_WIDTH,
                height=FLUX_IMAGE_HEIGHT,
                guidance_scale=0.0,   # schnell は CFG なし
            )
        result.images[0].save(filepath)
        print(f"[FLUX] 完了！保存先: {filepath}")
    except Exception as e:
        print(f"[FLUX] 生成エラー: {e}")


# ─────────────────────────────────────────────
# Gemma-4 画像分析（Transformers版）
# ─────────────────────────────────────────────
def analyze_image_with_gemma4(model, processor, image_path: str, user_query: str = "この画像について説明してください。"):
    """Gemma-4-E4B（Transformers版）で画像を分析"""
    try:
        if not os.path.exists(image_path):
            print(f"[画像分析] エラー: ファイルが見つかりません: {image_path}")
            return None
        
        print(f"[画像分析] 読み込み中: {image_path}")
        
        # 画像ファイルのサポート形式確認
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            print(f"[画像分析] エラー: サポートされていない画像形式です: {ext}")
            return None
        
        print(f"[画像分析] 前処理中...")
        
        # 画像を開く
        image = Image.open(image_path).convert("RGB")
        
        # チャット形式で処理
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_query}
            ]}
        ]
        
        # プロセッサで前処理
        inputs = processor(
            text=processor.apply_chat_template(conversation, add_generation_prompt=True),
            images=[image],
            return_tensors="pt"
        )
        
        print(f"[画像分析] 推論中...")
        
        # 推論（no_grad で メモリ節約）
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        
        # デコード
        result = processor.decode(output[0], skip_special_tokens=True)
        
        # チャットテンプレートのマークアップを削除
        if "<end_of_turn>" in result:
            result = result.split("<end_of_turn>")[-1].strip()
        
        return result
        
    except Exception as e:
        print(f"[画像分析] エラー: {e}")
        import traceback
        traceback.print_exc()
        return None




# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Llama-cpp AI Chat with Image Generation & Analysis")
    parser.add_argument("--model", type=str, default="qwen", choices=list(MODELS.keys()))
    parser.add_argument("--file", type=str, help="読み込むファイルのパスを指定してください")
    parser.add_argument("--enable-image", action="store_true",
                    help="Stable Diffusion による画像生成を有効にする")
    parser.add_argument("--enable-chilloutmix", action="store_true",
                    help="[×]chilloutmix_NiPrunedFp32Fixを読み込む");
    parser.add_argument("--enable-flux", action="store_true",
                    help="[×]FLUX.1-schnell による画像生成を有効にする（初回約 24GB DL）")
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
        sd_pipe = load_sd_pipeline(True)
    if args.enable_chilloutmix:
        sd_pipe = load_sd_pipeline(False)
        args.enable_image = True
    
    flux_pipe = None
    if args.enable_flux:
        flux_pipe = load_flux_pipeline()

    # テキストモデルロード
    selected = MODELS[args.model]
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"--- ロード中: {selected['name']} (n_ctx: {selected['n_ctx']}) ---")
    
    llm = None
    llm_gemma4 = None
    
    try:
        if args.model == "gemma4-multi":
            # Gemma-4-multi: Transformers版でロード（マルチモーダル対応）
            try:
                from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
                
                print(f"  → Transformers でマルチモーダルモデルとして読み込み中...")
                
                processor = AutoProcessor.from_pretrained(
                    selected["repo_id"],
                    trust_remote_code=True
                )
                
                # メモリ最適化: float16でロード（int8は未対応）
                model = AutoModelForCausalLM.from_pretrained(
                    selected["repo_id"],
                    dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True
                )
                
                llm_gemma4 = (model, processor)
                print(f"  → ロード完了（画像分析機能が利用可能です）")
                
            except Exception as e:
                print(f"  → エラー: {e}")
                print(f"  → 必要なパッケージをインストールしてください:")
                print(f"     pip install transformers accelerate")
                import traceback
                traceback.print_exc()
                return
        else:
            # その他のモデル: llama-cpp で読み込み
            n_batch = 512
            
            if "filename" not in selected:
                print(f"エラー: モデル設定が不正です")
                return
            
            if "repo_id" in selected:
                # HuggingFaceから取得
                llm = Llama.from_pretrained(
                    repo_id=selected["repo_id"],
                    filename=selected["filename"],
                    local_dir=MODEL_DIR,
                    offline=USE_LOCAL_MODEL,
                    verbose=False,
                    n_ctx=selected["n_ctx"],
                    n_batch=n_batch
                )
            else:
                # ローカルファイルから直接読み込み
                model_path = os.path.join(MODEL_DIR, selected["filename"])
                llm = Llama(
                    model_path=model_path,
                    verbose=False,
                    n_ctx=selected["n_ctx"],
                    n_batch=n_batch
                )
    except Exception as e:
        print(f"エラー: {e}")
        return

    history = []
    print(f"\n--- {selected['name']} とのチャット開始 ---")
    if args.model == "gemma4-multi":
        print(f"  {ANALYZE_PREFIX} <画像パス> [質問]                              → 画像を分析（デフォルト: 説明）")
        print(f"    例: {ANALYZE_PREFIX} ./photo.png この写真に何が写ってますか？")
        print(f"    例: {ANALYZE_PREFIX} ./chart.png グラフを解析してください")
    if args.enable_image:
        print(f"  {IMAGE_GEN_PREFIX} <英語プロンプト>                              → 画像を生成")
        print(f"  {IMAGE_GEN_PREFIX} <英語プロンプト> --neg <ネガティブプロンプト>  → ネガティブプロンプト付きで生成")
    elif args.enable_flux:
        print(f"  {FLUX_GEN_PREFIX} <英語プロンプト>  → FLUX.1-schnell で画像生成")
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
            body = user_input[len(IMAGE_GEN_PREFIX):].strip()
            if not body:
                print(f"例: {IMAGE_GEN_PREFIX} a beautiful mountain landscape at sunset")
                print(f"     ネガティブプロンプト指定: {IMAGE_GEN_PREFIX} <プロンプト> --neg <ネガティブプロンプト>")
                continue
            # --neg オプションを分割
            neg_sep = " --neg "
            if neg_sep in body:
                parts = body.split(neg_sep, 1)
                prompt = parts[0].strip()
                negative_prompt = parts[1].strip()
            else:
                prompt = body
                negative_prompt = ""
            generate_image(sd_pipe, prompt, IMAGE_OUTPUT_DIR, negative_prompt)
            continue
        elif user_input.startswith(ANALYZE_PREFIX):
            # Gemma-4-multi 画像分析
            if args.model != "gemma4-multi":
                print(f"[画像分析] このモデルは画像分析に対応していません。--model gemma4-multi を使用してください。")
                continue
            body = user_input[len(ANALYZE_PREFIX):].strip()
            if not body:
                print(f"例: {ANALYZE_PREFIX} ./photo.png")
                print(f"例: {ANALYZE_PREFIX} ./chart.png このグラフについて説明してください")
                continue
            # 画像パスと質問を分割
            parts = body.split(maxsplit=1)
            image_path = parts[0]
            user_query = parts[1] if len(parts) > 1 else "この画像について説明してください。"
            
            model, processor = llm_gemma4
            result = analyze_image_with_gemma4(model, processor, image_path, user_query)
            if result:
                print(f"\nAI ({selected['name']}): {result}")
            continue
        elif user_input.startswith(FLUX_GEN_PREFIX):
            if not args.enable_flux:
                print(f"[FLUX] 無効です。--enable-flux を付けて起動してください。")
                continue
            body = user_input[len(FLUX_GEN_PREFIX):].strip()
            if not body:
                print(f"例: {FLUX_GEN_PREFIX} a serene Japanese garden in golden hour light")
                continue
            generate_flux_image(flux_pipe, body, IMAGE_OUTPUT_DIR)
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
            if args.model == "gemma4-multi":
                # Gemma-4-multi: Transformers版
                model, processor = llm_gemma4
                
                # チャットテンプレートを適用
                text = processor.apply_chat_template(
                    history,
                    add_generation_prompt=True
                )
                
                inputs = processor(text=text, return_tensors="pt")
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                    )
                
                ai_answer = processor.decode(output[0], skip_special_tokens=True)
                
                # テンプレートマークアップを削除
                if "<end_of_turn>" in ai_answer:
                    ai_answer = ai_answer.split("<end_of_turn>")[-1].strip()
                if "<|turn>" in ai_answer:
                    ai_answer = ai_answer.split("model\n")[-1].strip()
            else:
                # その他のモデル: llama-cpp
                response = llm.create_chat_completion(messages=history, max_tokens=512)
                ai_answer = response["choices"][0]["message"]["content"]
            
            print(f"\nAI ({selected['name']}): {ai_answer}")
            history.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
