import asyncio
from playwright.async_api import async_playwright

# --- CONFIGURATION ---
# File where your Instagram login session will be saved
IG_AUTH_FILE = "ig_auth_state.json"

async def main():
    """
    This script logs into Instagram once and saves the authentication state.
    Later scrapers can reuse this state to appear as a logged-in user.
    """
    async with async_playwright() as p:
        # Non-headless so you can log in manually
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        print("\n--- Instagram Authentication Setup ---")
        print("A browser window will open now.")
        print("1. Please log in to your Instagram account.")
        print("2. Solve any CAPTCHAs or verification steps.")
        print("3. Once logged in and your feed is visible, CLOSE the browser window.")

        await page.goto("https://www.instagram.com/accounts/login/")

        # Wait for user to close browser tab after login
        print("\nWaiting for you to log in and close the browser. (No timeout)")
        await page.wait_for_event("close", timeout=0)

        # Save auth session
        await context.storage_state(path=IG_AUTH_FILE)

        print(f"\n✅ Instagram authentication state saved successfully to '{IG_AUTH_FILE}'!")
        print("You can now use this file in your Instagram scraper.")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
