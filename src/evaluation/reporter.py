import json
import os

def save_results(sonuclar):
    os.makedirs('artifacts/reports/', exist_ok=True)
  
    file_path = 'artifacts/reports/evaluation_results.json'
    with open(file_path, 'w') as f:
        json.dump(sonuclar, f, indent=4)
        
    print(f"Rapor başarıyla '{file_path}' adresine kaydedildi!")
