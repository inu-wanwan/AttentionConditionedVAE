import requests
from bs4 import BeautifulSoup
import csv
import os
from utils import load_config
from tqdm import tqdm

"""
scrape_dude_targets.py
Scrapes the DUD-E website and get pdb ids for each target.
output_file = dude_dir/targets_pdb_ids.csv
target name, pdb id, uniprot id
"""
def get_uniprot_id_from_pdb(pdb_id, entity_id=1):
    # URLを構築
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
    
    # APIリクエストを送信
    response = requests.get(url)
    
    # レスポンスが成功した場合
    if response.status_code == 200:
        data = response.json()
        uniprot_ids = []

        if "rcsb_polymer_entity_align" in data:
            for alignment in data["rcsb_polymer_entity_align"]:
                if alignment.get("reference_database_name") == "UniProt":
                    uniprot_ids.append(alignment.get("reference_database_accession"))
        
        return uniprot_ids[0]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

path_config = load_config('filepath.yml')

dude_dir = path_config['data']['DUD-E']
output_file = os.path.join(dude_dir, 'targets_pdb_ids.csv')

url = "http://dude.docking.org/targets"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')

targets = []

rows = soup.find_all('tr')

for row in tqdm(rows, desc="Scraping targets"):
    cols = row.find_all('td')
    if len(cols) >= 2:
        target_name = cols[1].get_text(strip=True)
        pdb_id = cols[2].get_text(strip=True)
        uniprot_id = get_uniprot_id_from_pdb(pdb_id)
        targets.append([target_name, pdb_id, uniprot_id])

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Target', 'PDB ID', 'UniProt ID'])
    writer.writerows(targets)

print(f"Scraped {len(targets)} targets to {output_file}")