import json
import csv
import os
import glob
from pathlib import Path

def decode_unicode(text):
    """Decode Unicode escape sequences and fix encoding issues"""
    if isinstance(text, str):
        try:
            # First try to decode Unicode escape sequences
            decoded = text.encode('utf-8').decode('unicode_escape')
            # Then fix the encoding issues where characters are showing as Ã + character
            decoded = decoded.replace('Ã¡', 'á')
            decoded = decoded.replace('Ã©', 'é')
            decoded = decoded.replace('Ã­', 'í')
            decoded = decoded.replace('Ã³', 'ó')
            decoded = decoded.replace('Ãº', 'ú')
            decoded = decoded.replace('Ã±', 'ñ')
            decoded = decoded.replace('Ã ', 'à')
            decoded = decoded.replace('Ã¨', 'è')
            decoded = decoded.replace('Ã¬', 'ì')
            decoded = decoded.replace('Ã²', 'ò')
            decoded = decoded.replace('Ã¹', 'ù')
            decoded = decoded.replace('Ã¼', 'ü')
            decoded = decoded.replace('Ã§', 'ç')
            decoded = decoded.replace('Ã«', 'ë')
            return decoded
        except:
            # If decoding fails, just fix the encoding issues
            text = text.replace('Ã¡', 'á')
            text = text.replace('Ã©', 'é')
            text = text.replace('Ã­', 'í')
            text = text.replace('Ã³', 'ó')
            text = text.replace('Ãº', 'ú')
            text = text.replace('Ã±', 'ñ')
            text = text.replace('Ã ', 'à')
            text = text.replace('Ã¨', 'è')
            text = text.replace('Ã¬', 'ì')
            text = text.replace('Ã²', 'ò')
            text = text.replace('Ã¹', 'ù')
            text = text.replace('Ã¼', 'ü')
            text = text.replace('Ã§', 'ç')
            text = text.replace('Ã«', 'ë')
            return text
    return text

def extract_suv_data(json_file_path):
    """Extract SUV data from a JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Check if it's an SUV
        if data.get('Carroceria') != 'SUV':
            return None
        
        # Handle Precio field safely (it can be null)
        precio_data = data.get('Precio') or {}
        
        # Extract the required fields
        suv_data = {
            'Marca': decode_unicode(data.get('Marca', '')),
            'Modelo': decode_unicode(data.get('Modelo', '')),
            'Version': decode_unicode(data.get('Version', '')),
            'VersionYear': data.get('VersionYear', ''),
            'Carroceria': decode_unicode(data.get('Carroceria', '')),
            'Segmento': decode_unicode(data.get('Segmento', '')),
            'Propulsion': decode_unicode(data.get('Propulsion', '')),
            'Transmision': decode_unicode(data.get('Transmision', '')),
            'Potencia': decode_unicode(data.get('Potencia', '')),
            'Cilindrada': decode_unicode(data.get('Cilindrada', '')),
            'Origen': decode_unicode(data.get('Origen', '')),
            'Garantia': decode_unicode(data.get('Garantia', '')),
            'Moneda': decode_unicode(precio_data.get('Moneda', '')),
            'Importe': precio_data.get('Importe', '')
        }
        
        return suv_data
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return None

def main():
    # Define the base directory and output file
    base_dir = Path('data/autocosmos')
    output_file = Path('data/suvs_autocosmos.csv')
    
    # Define the columns for the CSV
    columns = [
        'Marca', 'Modelo', 'Version', 'VersionYear', 'Carroceria', 
        'Segmento', 'Propulsion', 'Transmision', 'Potencia', 'Cilindrada', 
        'Origen', 'Garantia', 'Moneda', 'Importe'
    ]
    
    # Find all JSON files in the autocosmos directory
    json_files = []
    for year_dir in base_dir.glob('*'):
        if year_dir.is_dir():
            json_files.extend(year_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Extract SUV data
    suv_data_list = []
    suv_count = 0
    
    for json_file in json_files:
        suv_data = extract_suv_data(json_file)
        if suv_data:
            suv_data_list.append(suv_data)
            suv_count += 1
            print(f"Found SUV: {suv_data['Marca']} {suv_data['Modelo']} {suv_data['Version']}")
    
    print(f"\nTotal SUVs found: {suv_count}")
    
    if suv_data_list:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(suv_data_list)
        
        print(f"\nCSV file created successfully: {output_file}")
        print(f"Total records written: {len(suv_data_list)}")
    else:
        print("No SUV data found to write to CSV")

if __name__ == "__main__":
    main()
