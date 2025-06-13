from playwright.sync_api import sync_playwright
import time
import json
import random
import pandas as pd
from datetime import datetime
import os

def scrape_mercadolibre_suvs(headless=False, save_format="json"):
    """
    Scrape SUVs data from MercadoLibre.
    
    Args:
        headless (bool): Whether to run browser in headless mode
        save_format (str): Format to save data ("json" or "csv")
    """
    # Create a directory for the data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # URL of the MercadoLibre SUVs page
    url = "https://autos.mercadolibre.com.ar/_NoIndex_True_VEHICLE*BODY*TYPE_452759"
    
    with sync_playwright() as p:
        # Launch the browser with a realistic user agent
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            viewport={'width': 1366, 'height': 768}
        )
        
        # Create a new page
        page = context.new_page()
        
        try:
            # Navigate to the URL
            print("Navigating to MercadoLibre...")
            page.goto(url)
            
            # Wait for the content to load
            page.wait_for_selector('.ui-search-layout__item', timeout=30000)
            
            # Scroll down the page to load more items (lazy loading)
            print("Scrolling to load more items...")
            for _ in range(5):  # Adjust number of scrolls as needed
                page.evaluate("window.scrollBy(0, 800)")  # Scroll down by 800 pixels
                time.sleep(0.5 + random.random())  # Random delay between 0.5-1.5 seconds
            
            # Wait for additional content to load
            time.sleep(2)
            
            # Extract all SUV items
            items = page.query_selector_all('.ui-search-layout__item')
            
            suvs_data = []
            
            for item in items:
                try:
                    # Extract title using the new selector
                    title_element = item.query_selector('h3.poly-component__title-wrapper a.poly-component__title')
                    title = title_element.inner_text() if title_element else 'N/A'
                    
                    # Extract price using the new selector
                    price_element = item.query_selector('.poly-component__price .andes-money-amount__fraction')
                    price = price_element.inner_text() if price_element else 'N/A'
                    
                    # Extract year and kilometers
                    attributes_list = item.query_selector('.poly-component__attributes-list')
                    year = 'N/A'
                    kilometers = 'N/A'
                    
                    if attributes_list:
                        attributes = attributes_list.query_selector_all('.poly-attributes_list__item')
                        if len(attributes) >= 2:
                            year = attributes[0].inner_text()
                            kilometers = attributes[1].inner_text()
                    
                    # Extract link
                    link = title_element.get_attribute('href') if title_element else 'N/A'
                    
                    # Extract location if available
                    location = item.query_selector('.ui-search-item__location')
                    location_text = location.inner_text() if location else 'N/A'
                    
                    suv_data = {
                        'title': title,
                        'price': price,
                        'year': year,
                        'kilometers': kilometers,
                        'location': location_text,
                        'link': link,
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    suvs_data.append(suv_data)
                    print(f"Scraped: {title} - {price} - {year} - {kilometers}")
                    
                except Exception as e:
                    print(f"Error extracting item data: {str(e)}")
                    continue
            
            
            # Save the data in the specified format
            if save_format.lower() == "csv":
                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(suvs_data)
                filename = f'data/mercadolibre_suvs.csv'
                df.to_csv(filename, index=False, encoding='utf-8')

            
            print(f"Successfully scraped {len(suvs_data)} SUVs. Data saved to {filename}")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        finally:
            # Close the browser
            browser.close()

if __name__ == "__main__":
    scrape_mercadolibre_suvs(headless=False, save_format="csv")
