
#import requests # Keep this to test after the fix
import requests
import os
from dotenv import load_dotenv
from flask import Flask, app

# Load environment variables
load_dotenv()
app = Flask(__name__)
base_url = os.getenv("URL_FOR_API")
# Make a GET request to the REST API
@app.route('/country_info/<country_name>', methods=['GET'])
def get_country_info(country_name):
    url = base_url + country_name
    print(f"Making request to base URL: {url}")
    response = requests.get(url)
    output = response.json()
    official_name = output[0]['name']['official']
    # Print the response
    if response.status_code == 200:
        print(f"Official Name: {official_name}")
        return official_name
    else:
        print(f"Error: {response.status_code}") 