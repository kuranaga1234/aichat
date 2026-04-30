
# -------------------------------------------------
# まず、必要なライブラリをインストールし、ブラウザの実体をセットアップします。
# pip install playwright beautifulsoup4
# playwright install chromium


# -------------------------------------------------
# まず、必要なライブラリをインストールし、ブラウザの実体をセットアップします。
# python script.py https://www.google.com "Google"


# -------------------------------------------------
# このスクリプトは、引数でURLを受け取り、指定された一連の動作を実行します。
import sys
import os
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def analyze_web_page(url, search_text):
    # 保存ファイル名の設定
    screenshot_path = "screenshot.png"

    with sync_playwright() as p:
        # Chrome(Chromium)を起動
        # headless=Falseにすると実際にブラウザが動く様子が見えます
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        try:
            print(f"URLにアクセス中: {url}")
            # ページを開き、ネットワークが安定する（待機処理）まで待つ
            page.goto(url, wait_until="networkidle")

            # 1. スクリーンショットの保存
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"スクリーンショットを保存しました: {screenshot_path}")

            # 2. HTMLの解析
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')

            # 3. 特定の文字列があるかどうかを判定
            # soup.get_text()でHTMLタグを除いた純粋なテキストから検索します
            if search_text in soup.get_text():
                result = f"結果: 指定した文字列 '{search_text}' は見つかりました。"
            else:
                result = f"結果: 指定した文字列 '{search_text}' は見つかりませんでした。"

            print(result)
            return result

        except Exception as e:
            error_msg = f"エラーが発生しました: {e}"
            print(error_msg)
            return error_msg
        finally:
            browser.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使い方: python script.py [URL] [検索したい文字列]")
        sys.exit(1)

    target_url = sys.argv[1]
    keyword = sys.argv[2]
    
    analyze_web_page(target_url, keyword)



