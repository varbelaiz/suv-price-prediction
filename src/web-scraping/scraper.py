from playwright.sync_api import sync_playwright
import time
import json
import random
import pandas as pd
from datetime import datetime
import os
import re

def extract_quantity_from_results_qty(results_qty):
    """
    Extract numeric quantity from results_qty string like "(924)" or "924 resultados"
    
    Args:
        results_qty (str): String containing quantity information
        
    Returns:
        int: Extracted quantity number, or 0 if not found
    """
    if not results_qty or results_qty == 'N/A':
        return 0
    
    # Remove parentheses and extract numbers
    numbers = re.findall(r'\d+', str(results_qty))
    if numbers:
        return int(numbers[0])
    return 0

def scrape_grid_items(page, grid_selector='.ui-search-search-modal-grid-columns .ui-search-search-modal-filter'):
    """
    Generic function to scrape items from a grid (works for both brands and models).
    
    Args:
        page: Playwright page object
        grid_selector: CSS selector for grid items
    
    Returns:
        list: List of dictionaries containing item data
    """
    items_data = []
    
    # Extract all items from the grid
    item_elements = page.query_selector_all(grid_selector)
    
    for i, item_element in enumerate(item_elements):
        try:
            # Extract item name
            item_name_element = item_element.query_selector('.ui-search-search-modal-filter-name')
            item_name = item_name_element.inner_text() if item_name_element else 'N/A'
            
            # Extract item link
            item_link = item_element.get_attribute('href')
            
            # Extract title attribute if available
            item_title = item_element.get_attribute('title')
            
            item_data = {
                'name': item_name,
                'title': item_title,
                'link': item_link,
            }
            
            items_data.append(item_data)
            
        except Exception as e:
            print(f"Error extracting item data for item {i+1}: {str(e)}")
            continue
    
    return items_data

def scrape_versions_for_model(page, brand_name, model_name, model_link):
    """
    Scrape all versions for a specific model.
    
    Args:
        page: Playwright page object
        brand_name: Name of the brand
        model_name: Name of the model
        model_link: URL of the model page
    
    Returns:
        list: List of version data for the model
    """
    try:
        print(f"    Navigating to model page: {brand_name} {model_name}")
        page.goto(model_link)
        
        # Wait for the page to load
        time.sleep(1 + random.random())
        
        # Look for the "Versiones" section
        versiones_sections = page.query_selector_all('.ui-search-filter-dl')
        versiones_section = None
        
        for section in versiones_sections:
            # Check if this section contains the "Versiones" title
            title_element = section.query_selector('h3.ui-search-filter-dt-title')
            if title_element and 'Versiones' in title_element.inner_text():
                versiones_section = section
                break
        
        if not versiones_section:
            print(f"      No 'Versiones' section found for model: {brand_name} {model_name}")
            return []
        
        # Check if there's a "Mostrar más" link (case 1: grid)
        show_more_link = versiones_section.query_selector('a.ui-search-modal__link[title="Mostrar más"]')
        
        if show_more_link:
            # Case 1: Navigate to grid page
            versiones_link = show_more_link.get_attribute('href')
            print(f"      Found 'Mostrar más' link for {brand_name} {model_name} versions, navigating to grid...")
            page.goto(versiones_link)
            
            # Wait for the versions grid to load
            page.wait_for_selector('.ui-search-search-modal-grid-columns', timeout=15000)
            time.sleep(1 + random.random())
            
            # Scrape the versions using the generic function
            versions_data = scrape_grid_items(page)
        else:
            # Case 2: Versions are directly listed in the current page
            print(f"      Versions are listed directly on page for {brand_name} {model_name}")
            versions_data = []
            
            # Extract versions from the list items in the ul
            version_items = versiones_section.query_selector_all('ul li.ui-search-filter-container')
            
            for version_item in version_items:
                try:
                    # Extract version link
                    version_link_element = version_item.query_selector('a.ui-search-link')
                    version_link = version_link_element.get_attribute('href') if version_link_element else 'N/A'
                    
                    # Extract version name
                    version_name_element = version_item.query_selector('.ui-search-filter-name')
                    version_name = version_name_element.inner_text() if version_name_element else 'N/A'
                    
                    # Extract version title (from the link's title attribute)
                    version_title = version_link_element.get_attribute('title') if version_link_element else 'N/A'
                    
                    # Extract results quantity (optional)
                    results_qty_element = version_item.query_selector('.ui-search-filter-results-qty')
                    results_qty = results_qty_element.inner_text() if results_qty_element else '(0)'
                    
                    version_data = {
                        'name': version_name,
                        'title': version_title,
                        'link': version_link,
                        'results_qty': results_qty
                    }
                    
                    versions_data.append(version_data)
                    
                except Exception as e:
                    print(f"        Error extracting version data: {str(e)}")
                    continue
        
        # Add brand and model information to each version
        for version in versions_data:
            version['brand_name'] = brand_name
            version['model_name'] = model_name
            version['scraped_at'] = datetime.now().isoformat()
        
        print(f"      Found {len(versions_data)} versions for {brand_name} {model_name}")
        return versions_data
        
    except Exception as e:
        print(f"      Error scraping versions for {brand_name} {model_name}: {str(e)}")
        return []

def scrape_models_for_brand(page, brand_name, brand_link):
    """
    Scrape all models for a specific brand.
    
    Args:
        page: Playwright page object
        brand_name: Name of the brand
        brand_link: URL of the brand page
    
    Returns:
        list: List of model data for the brand
    """
    try:
        print(f"  Navigating to brand page: {brand_name}")
        page.goto(brand_link)
        
        # Wait for the page to load
        time.sleep(2 + random.random())
        
        # Look for the "Modelo" section
        modelo_sections = page.query_selector_all('.ui-search-filter-dl')
        modelo_section = None
        
        for section in modelo_sections:
            # Check if this section contains the "Modelo" title
            title_element = section.query_selector('h3.ui-search-filter-dt-title')
            if title_element and 'Modelo' in title_element.inner_text():
                modelo_section = section
                break
        
        if not modelo_section:
            print(f"    No 'Modelo' section found for brand: {brand_name}")
            return []
        
        # Check if there's a "Mostrar más" link (case 1: grid)
        show_more_link = modelo_section.query_selector('a.ui-search-modal__link[title="Mostrar más"]')
        
        if show_more_link:
            # Case 1: Navigate to grid page
            modelo_link = show_more_link.get_attribute('href')
            print(f"    Found 'Mostrar más' link for {brand_name}, navigating to grid...")
            page.goto(modelo_link)
            
            # Wait for the models grid to load
            page.wait_for_selector('.ui-search-search-modal-grid-columns', timeout=15000)
            time.sleep(1 + random.random())
            
            # Scrape the models using the generic function
            models_data = scrape_grid_items(page)
        else:
            # Case 2: Models are directly listed in the current page
            print(f"    Models are listed directly on page for {brand_name}")
            models_data = []
            
            # Extract models from the list items in the ul
            model_items = modelo_section.query_selector_all('ul li.ui-search-filter-container')
            
            for model_item in model_items:
                try:
                    # Extract model link
                    model_link_element = model_item.query_selector('a.ui-search-link')
                    model_link = model_link_element.get_attribute('href') if model_link_element else 'N/A'
                    
                    # Extract model name
                    model_name_element = model_item.query_selector('.ui-search-filter-name')
                    model_name = model_name_element.inner_text() if model_name_element else 'N/A'
                    
                    # Extract model title (from the link's title attribute)
                    model_title = model_link_element.get_attribute('title') if model_link_element else 'N/A'
                    
                    # Extract results quantity (optional)
                    results_qty_element = model_item.query_selector('.ui-search-filter-results-qty')
                    results_qty = results_qty_element.inner_text() if results_qty_element else '(0)'
                    
                    model_data = {
                        'name': model_name,
                        'title': model_title,
                        'link': model_link,
                        'results_qty': results_qty
                    }
                    
                    models_data.append(model_data)
                    
                except Exception as e:
                    print(f"      Error extracting model data: {str(e)}")
                    continue
        
        # Add brand information to each model and scrape versions
        for model in models_data:
            model['brand_name'] = brand_name
            model['scraped_at'] = datetime.now().isoformat()
            
            # Scrape versions for this model
            if model['link'] and model['link'] != 'N/A':
                versions = scrape_versions_for_model(page, brand_name, model['name'], model['link'])
                model['versions'] = versions
            else:
                model['versions'] = []
        
        print(f"    Found {len(models_data)} models for {brand_name}")
        return models_data
        
    except Exception as e:
        print(f"    Error scraping models for {brand_name}: {str(e)}")
        return []

def scrape_mercadolibre_brands_models_and_versions(headless=False):
    """
    Scrape all Brands, their Models, and Versions from MercadoLibre SUVs filter page.
    Saves directly as CSV with Brand, Model, Version, Quantity columns.
    
    Args:
        headless (bool): Whether to run browser in headless mode
    """
    # Create a directory for the data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # URL of the MercadoLibre Brands filter page
    url = "https://listado.mercadolibre.com.ar/suvs_FiltersAvailableSidebar?filter=BRAND&sb=all_mercadolibre"
    
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
            print("Navigating to MercadoLibre Brands filter page...")
            page.goto(url)
            
            # Wait for the content to load
            page.wait_for_selector('.ui-search-search-modal-grid-columns', timeout=30000)
            
            # Wait a bit for the page to fully load
            time.sleep(2)
            
            # Extract all brand elements from the grid using the generic function
            brands_data = scrape_grid_items(page)
            
            print(f"Found {len(brands_data)} brands to scrape...")
            
            # Now scrape models for each brand
            all_brands_and_models = []
            
            for i, brand_data in enumerate(brands_data):
                brand_name = brand_data['name']
                brand_link = brand_data['link']
                
                print(f"Scraping brand {i+1}/{len(brands_data)}: {brand_name}")
                
                # Add brand info to the result
                brand_info = {
                    'brand_name': brand_name,
                    'brand_title': brand_data['title'],
                    'brand_link': brand_link,
                    'models': [],
                    'scraped_at': datetime.now().isoformat()
                }
                
                # Scrape models for this brand
                if brand_link and brand_link != 'N/A':
                    models = scrape_models_for_brand(page, brand_name, brand_link)
                    brand_info['models'] = models
                
                all_brands_and_models.append(brand_info)
                
                # Add a small delay between brands to be respectful
                time.sleep(1 + random.random())
            
            # Convert to CSV format with Brand, Model, Version, Quantity columns
            csv_rows = []
            
            # Process each brand
            for brand in all_brands_and_models:
                brand_name = brand['brand_name']
                
                # Check if brand has models
                if not brand.get('models'):
                    # Brand with no models
                    csv_rows.append({
                        'Brand': brand_name,
                        'Model': 'N/A',
                        'Version': 'N/A',
                        'Quantity': 0
                    })
                    continue
                
                # Process each model in the brand
                for model in brand['models']:
                    model_name = model.get('name', 'N/A')
                    
                    # Check if model has versions
                    if not model.get('versions'):
                        # Model with no versions
                        csv_rows.append({
                            'Brand': brand_name,
                            'Model': model_name,
                            'Version': 'N/A',
                            'Quantity': 0
                        })
                        continue
                    
                    # Process each version in the model
                    for version in model['versions']:
                        version_name = version.get('name', 'N/A')
                        results_qty = version.get('results_qty', '(0)')
                        
                        # Extract quantity number from results_qty
                        quantity = extract_quantity_from_results_qty(results_qty)
                        
                        csv_rows.append({
                            'Brand': brand_name,
                            'Model': model_name,
                            'Version': version_name,
                            'Quantity': quantity
                        })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(csv_rows)
            
            # Sort by Brand, Model, then by Quantity (descending)
            df = df.sort_values(['Brand', 'Model', 'Quantity'], ascending=[True, True, False])
            
            # Save to CSV
            filename = f'data/mercadolibre_versions.csv'
            df.to_csv(filename, index=False, encoding='utf-8')
            
            print(f"Successfully scraped and saved data to {filename}")
            print(f"Total rows: {len(df)}")
            print(f"Total brands: {df['Brand'].nunique()}")
            print(f"Total models: {df[df['Model'] != 'N/A']['Model'].nunique()}")
            print(f"Total versions: {df[df['Version'] != 'N/A']['Version'].nunique()}")
            print(f"Total quantity across all versions: {df['Quantity'].sum():,}")
            
            # Show sample data
            print("\nFirst 10 rows:")
            print(df.head(10).to_string(index=False))
            
            # Show top versions by quantity
            print("\nTop 10 versions by quantity:")
            top_versions = df[df['Version'] != 'N/A'].nlargest(10, 'Quantity')
            print(top_versions.to_string(index=False))
            
            return df
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        
        finally:
            # Close the browser
            browser.close()

if __name__ == "__main__":
    result = scrape_mercadolibre_brands_models_and_versions(headless=False)
