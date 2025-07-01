import requests
import json
import hashlib
import hmac
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode


class AutocosmosAPI:
    """
    A class to interact with the Autocosmos API for car version matching.
    """
    
    def __init__(self, app_key: str, app_secret: str, base_url: str = "https://api.autocosmos.com.mx"):
        """
        Initialize the Autocosmos API client.
        
        Args:
            app_key (str): Your Autocosmos application key
            app_secret (str): Your Autocosmos application secret
            base_url (str): The base URL for the Autocosmos API
        """
        self.base_url = base_url.rstrip('/')
        self.app_key = app_key
        self.app_secret = app_secret
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def _generate_canonical_string(self, method: str, digest: Optional[str], date_str: str, 
                                 acs_headers: Dict[str, str], path_query: str) -> str:
        """
        Generate the canonical string for HMAC calculation as per Autocosmos API docs.
        
        Args:
            method (str): HTTP method (GET, POST, PUT, etc.)
            digest (str, optional): Digest header value
            date_str (str): Date header value
            acs_headers (Dict[str, str]): X-ACS- headers
            path_query (str): Path and query string
            
        Returns:
            str: Canonical string for HMAC calculation
        """
        parts = []
        parts.append(method)
        parts.append(digest if digest else '')
        parts.append(date_str if date_str else '')
        
        # Add X-ACS- headers in canonical format
        if acs_headers:
            # Sort headers by lowercase name
            sorted_headers = sorted(acs_headers.items(), key=lambda x: x[0].lower())
            for header_name, header_value in sorted_headers:
                # Lowercase header name and trim values
                canonical_header = f"{header_name.lower().strip()}:{header_value.strip()}"
                parts.append(canonical_header)
        
        parts.append(path_query)
        
        # Join with newlines
        return '\n'.join(parts)
    
    def _calculate_hmac(self, canonical_string: str) -> str:
        """
        Calculate HMAC-SHA256 for the canonical string.
        
        Args:
            canonical_string (str): The canonical string to hash
            
        Returns:
            str: Base64 encoded HMAC
        """
        # Convert canonical string to UTF-8 bytes
        message = canonical_string.encode('utf-8')
        key = self.app_secret.encode('utf-8')
        
        # Calculate HMAC-SHA256
        hmac_obj = hmac.new(key, message, hashlib.sha256)
        return base64.b64encode(hmac_obj.digest()).decode('utf-8')
    
    def _generate_auth_headers(self, method: str, path_query: str, payload: Optional[str] = None) -> Dict[str, str]:
        """
        Generate authentication headers for the request.
        
        Args:
            method (str): HTTP method
            path_query (str): Path and query string
            payload (str, optional): Request payload for digest calculation
            
        Returns:
            Dict[str, str]: Headers dictionary
        """
        headers = {}
        
        # Calculate digest if there's a payload
        digest = None
        if payload:
            digest_hash = hashlib.sha256(payload.encode('utf-8')).digest()
            digest = f"sha-256={base64.b64encode(digest_hash).decode('utf-8')}"
            headers['Digest'] = digest
        
        # Generate date
        date_str = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        headers['Date'] = date_str
        
        # X-ACS- headers (empty for now, but can be extended)
        acs_headers = {}
        
        # Generate canonical string
        canonical_string = self._generate_canonical_string(
            method, digest, date_str, acs_headers, path_query
        )
        
        # Calculate HMAC
        hmac_value = self._calculate_hmac(canonical_string)
        
        # Set Authorization header
        headers['Authorization'] = f'ACS-HMAC {self.app_key}:{hmac_value}'
        
        return headers
    
    def find_car_version(
        self,
        make: str,
        model: str,
        trim: Optional[str] = None,
        year: Optional[int] = None,
        bodystyle: Optional[str] = None,
        transmission_type: Optional[str] = None,
        displacement: Optional[str] = None,
        power: Optional[str] = None,
        fuel_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find the best matching car version using the Autocosmos API.
        
        Args:
            make (str): Brand/make of the car (required)
            model (str): Model of the car (required)
            trim (str, optional): Trim/version of the car
            year (int, optional): Manufacturing year
            bodystyle (str, optional): Body style (e.g., "coupe", "sedan", "suv")
            transmission_type (str, optional): Transmission type ("A", "Automatico", "M", "Manual")
            displacement (str, optional): Engine displacement (e.g., "2.0", "2.8L")
            power (str, optional): Engine power (e.g., "156", "204cv")
            fuel_type (str, optional): Fuel type (e.g., "gasolina", "diesel")
        
        Returns:
            Dict[str, Any]: API response containing match information and candidates
            
        Raises:
            requests.RequestException: If the API request fails
            ValueError: If required parameters are missing
        """
        if not make or not model:
            raise ValueError("Both 'make' and 'model' are required parameters")
        
        # Build query parameters
        params = {
            'make': make,
            'model': model
        }
        
        # Add optional parameters if provided
        if trim:
            params['trim'] = trim
        if year:
            params['year'] = year
        if bodystyle:
            params['bodystyle'] = bodystyle
        if transmission_type:
            params['transmissionType'] = transmission_type
        if displacement:
            params['displacement'] = displacement
        if power:
            params['power'] = power
        if fuel_type:
            params['fuelType'] = fuel_type
        
        # Construct the path and query
        path_query = f"/v3/versiones/bestmatch?{urlencode(params)}"
        
        # Generate authentication headers
        auth_headers = self._generate_auth_headers('GET', path_query)
        
        # Make the request
        url = f"{self.base_url}{path_query}"
        
        try:
            response = self.session.get(url, headers=auth_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"API request failed: {str(e)}")
    
    def get_version_details(self, version_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific car version.
        
        Args:
            version_id (str): The version ID from the bestmatch response
            
        Returns:
            Dict[str, Any]: Detailed version information
        """
        path_query = f"/v3/versiones/{version_id}"
        auth_headers = self._generate_auth_headers('GET', path_query)
        url = f"{self.base_url}{path_query}"
        
        try:
            response = self.session.get(url, headers=auth_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"Failed to get version details: {str(e)}")


def find_car_version_simple(
    make: str,
    model: str,
    app_key: str,
    app_secret: str,
    fuel_type: Optional[str] = None,
    transmission_type: Optional[str] = None,
    year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simple function to find car version with basic parameters.
    
    Args:
        make (str): Brand/make of the car
        model (str): Model of the car
        app_key (str): Your Autocosmos application key
        app_secret (str): Your Autocosmos application secret
        fuel_type (str, optional): Fuel type (e.g., "gasolina", "diesel")
        transmission_type (str, optional): Transmission type ("A", "Automatico", "M", "Manual")
        year (int, optional): Manufacturing year
    
    Returns:
        Dict[str, Any]: API response with match information
    """
    api = AutocosmosAPI(app_key, app_secret)
    return api.find_car_version(
        make=make,
        model=model,
        fuel_type=fuel_type,
        transmission_type=transmission_type,
        year=year
    )


# Example usage and utility functions
def print_match_info(response: Dict[str, Any]) -> None:
    """
    Print formatted information about the car version match.
    
    Args:
        response (Dict[str, Any]): Response from find_car_version
    """
    print("=== Car Version Match Results ===")
    
    if response.get('Match'):
        match = response['Match']
        print(f"✅ Found exact match!")
        print(f"Brand: {match.get('Marca', 'N/A')}")
        print(f"Model: {match.get('Modelo', 'N/A')}")
        print(f"Version: {match.get('Nombre', 'N/A')}")
        print(f"Body Style: {match.get('Carroceria', 'N/A')}")
        print(f"Fuel Type: {match.get('Propulsion', 'N/A')}")
        print(f"Transmission: {match.get('Transmision', 'N/A')}")
        print(f"Power: {match.get('Potencia', 'N/A')}")
        print(f"Displacement: {match.get('Cilindrada', 'N/A')}")
        print(f"Segment: {match.get('Segmento', 'N/A')}")
        print(f"Version ID: {match.get('_links', {}).get('href', 'N/A')}")
    else:
        print("❌ No exact match found")
        print(f"Make: {response.get('Make', 'N/A')}")
        print(f"Model: {response.get('Model', 'N/A')}")
        print(f"Trim: {response.get('Trim', 'N/A')}")
        print(f"Fuel Type: {response.get('FuelType', 'N/A')}")
        print(f"Transmission Type: {response.get('TransmisionType', 'N/A')}")
        
        candidates = response.get('Candidates', [])
        if candidates:
            print(f"\n📋 Found {len(candidates)} candidate(s):")
            for i, candidate in enumerate(candidates, 1):
                print(f"  {i}. {candidate.get('Marca', 'N/A')} {candidate.get('Modelo', 'N/A')} - {candidate.get('Nombre', 'N/A')}")


if __name__ == "__main__":
    # Example usage - YOU NEED TO PROVIDE YOUR OWN API CREDENTIALS

    
    # Replace these with your actual API credentials
    APP_KEY = "f8dfdd49902945d2a0d9135836f4732c"
    APP_SECRET = "1cb27d908ff24e51bdf39ee14ce2e72f"
    

    try:
        api = AutocosmosAPI(APP_KEY, APP_SECRET)
        result2 = api.find_car_version(
            make="Ferrari",
            model="488",
        )
        print("Example 2: BMW Serie 3 Executive Coupe")
        print_match_info(result2)
            
    except Exception as e:

        print(f"Error: {e}")
