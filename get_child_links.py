import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_child_links(parent_url):
    child_links = []

    try:
        response = requests.get(parent_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            child_url = urljoin(parent_url, link['href'])
            child_links.append(child_url)

    except requests.exceptions.RequestException as e:
        print("Error:", e)

    return child_links

parent_website = "https://www.americanexpress.com"
child_links = get_child_links(parent_website)

for idx, child_link in enumerate(child_links, start=1):
    print(f"{idx}. {child_link}")
