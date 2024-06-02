import requests

def fetch_html_content(url):
    """
    Fetch the HTML content of a web URL.

    Parameters:
    url (str): The URL of the webpage to fetch.

    Returns:
    str: The HTML content of the webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None