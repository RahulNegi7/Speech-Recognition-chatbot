"""
Captures authentic screenshots of the running Streamlit chatbot application
using Playwright and local Google Chrome.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

screenshots_dir = Path(__file__).parent / "screenshots"
screenshots_dir.mkdir(parents=True, exist_ok=True)

chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def capture_screenshots():
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(
            executable_path=chrome_path,
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(viewport={"width": 1366, "height": 850})
        page = context.new_page()

        print("Navigating to Streamlit app at http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(3)  # Allow Streamlit React components to stabilize

        # 1. Capture App Overview
        overview_path = screenshots_dir / "01_app_overview.png"
        page.screenshot(path=str(overview_path))
        print(f"Captured: {overview_path}")

        # 2. Click a sample question button to show a real interaction
        sample_buttons = page.locator("button:has-text('What is the difference between LSTM and GRU?')")
        if sample_buttons.count() > 0:
            print("Clicking sample question: 'What is the difference between LSTM and GRU?'...")
            sample_buttons.first.click()
            # Wait for model generation to complete and text to render
            print("Waiting for NLP model inference...")
            time.sleep(8)
            answered_path = screenshots_dir / "02_question_answered.png"
            page.screenshot(path=str(answered_path))
            print(f"Captured: {answered_path}")

        # 3. Switch to Voice Input tab and show ASR interface
        voice_tab = page.locator("button:has-text('Voice Input')")
        if voice_tab.count() > 0:
            print("Switching to Voice Input tab...")
            voice_tab.first.click()
            time.sleep(1)
            audio_tab_path = screenshots_dir / "03_voice_input_tab.png"
            page.screenshot(path=str(audio_tab_path))
            print(f"Captured: {audio_tab_path}")

        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    capture_screenshots()

