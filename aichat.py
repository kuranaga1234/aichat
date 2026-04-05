import os
import argparse
from llama_cpp import Llama

# --- 設定定数 ---
USE_LOCAL_MODEL = True
MODEL_DIR = "./models"
MODELS = {
    "qwen": {"name": "Qwen2.5-7B", "repo_id": "Bartowski/Qwen2.5-7B-Instruct-GGUF", "filename": "*Q4_K_M.gguf"},
    "gemma": {"name": "Gemma-2-9B", "repo_id": "Bartowski/gemma-2-9b-it-GGUF", "filename": "*Q4_K_M.gguf"},
    "elyza": {"name": "Llama-3-ELYZA-JP-8B", "repo_id": "elyza/Llama-3-ELYZA-JP-8B-GGUF", "filename": "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"}
}

def main():
    parser = argparse.ArgumentParser(description="Llama-cpp AI Chat with File Support")
    parser.add_argument("--model", type=str, default="qwen", choices=list(MODELS.keys()))
    # ファイル指定用の引数を追加
    parser.add_argument("--file", type=str, help="読み込むファイルのパスを指定してください")
    args = parser.parse_args()

    # --- ファイルの中身を読み取る処理 ---
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

    selected = MODELS[args.model]
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    llm = Llama.from_pretrained(
        repo_id=selected["repo_id"],
        filename=selected["filename"],
        local_dir=MODEL_DIR,
        offline=USE_LOCAL_MODEL,
        verbose=False,
        n_ctx=4096 # ファイルを読み込むため、少し大きめに設定
    )

    history = []
    print(f"--- {selected['name']} とのチャット開始 ---")

    while True:
        user_input = input("\nあなた: ")
        if user_input.lower() in ['exit', 'quit', '終了']: break

        prompt = user_input
        # 初回のみ、ファイルの中身をコンテキストとして注入
        if len(history) == 0:
            instruction = "あなたは優秀なアシスタントです。提供されたファイルの内容を元に回答してください。\n\n"
            if file_content:
                prompt = f"{instruction}[ファイルの内容]:\n{file_content}\n\n[ユーザーの質問]: {user_input}"
            else:
                prompt = f"{instruction}[ユーザーの質問]: {user_input}"

        history.append({"role": "user", "content": prompt})

        try:
            response = llm.create_chat_completion(messages=history)
            ai_answer = response["choices"][0]["message"]["content"]
            print(f"\nAI ({selected['name']}): {ai_answer}")
            history.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            break

if __name__ == "__main__":
    main()