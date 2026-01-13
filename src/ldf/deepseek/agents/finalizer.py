import os
import json
import csv
from typing import List, Dict, Any

from ldf.deepseek.agents.base import Agent

class FinalizerAgent(Agent):
    """
    Agent 4: Finalizer
    Role: Save output.
    """
    def __init__(self):
        super().__init__("Finalizer")

    def process(self, data: Any, output_dir: str):
        self.log(f"Saving final output to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save JSON
        out_path_json = os.path.join(output_dir, "final_consolidated_output.json")
        with open(out_path_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.log(f"Saved JSON: {out_path_json}")
        
        # 2. Save CSV (Flattened)
        out_path_csv = os.path.join(output_dir, "final_consolidated_output.csv")
        try:
            # Determine max columns dynamically or fixed?
            # Prompt asked for Column I, II, III.
            headers = ["Minister", "Row_ID", "Column I", "Column II", "Column III"]
            
            with open(out_path_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for entry in data:
                    minister = entry.get("Minister", "Unknown")
                    
                    # Align columns by index
                    col1 = entry.get("Column I", [])
                    col2 = entry.get("Column II", [])
                    col3 = entry.get("Column III", [])
                    
                    # Ensure they are lists
                    if isinstance(col1, str): col1 = [col1]
                    if isinstance(col2, str): col2 = [col2]
                    if isinstance(col3, str): col3 = [col3]
                    
                    # Get max length to iterate rows
                    max_len = max(len(col1), len(col2), len(col3))
                    
                    for i in range(max_len):
                        val1 = col1[i] if i < len(col1) else ""
                        val2 = col2[i] if i < len(col2) else ""
                        val3 = col3[i] if i < len(col3) else ""
                        
                        writer.writerow([minister, i+1, val1, val2, val3])
                        
            self.log(f"Saved CSV: {out_path_csv}")
            
        except Exception as e:
            self.log(f"Error saving CSV: {e}")

        # 3. Create Per-Minister Folders
        self.log("Creating per-minister output folders...")
        try:
            for entry in data:
                minister_name = entry.get("Minister", "Unknown")
                
                # Snake Case Conversion
                # 1. Remove titles (Hon., etc) for folder name clarity
                clean_for_folder = minister_name.replace("Hon.", "").replace("Dr.", "").replace("Mr.", "")
                # 2. Lowercase and replace spaces/symbols with _
                safe_name = "".join([c if c.isalnum() else '_' for c in clean_for_folder]).strip('_')
                # 3. Collapse multiple underscores
                while "__" in safe_name:
                    safe_name = safe_name.replace("__", "_")
                safe_name = safe_name.lower()
                
                if not safe_name: safe_name = "unknown_minister"
                
                minister_dir = os.path.join(output_dir, safe_name)
                os.makedirs(minister_dir, exist_ok=True)
                
                # A. Save Minister JSON
                min_json_path = os.path.join(minister_dir, "data.json")
                with open(min_json_path, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
                    
                # B. Save Minister CSV
                min_csv_path = os.path.join(minister_dir, "data.csv")
                headers = ["Minister", "Row_ID", "Column I", "Column II", "Column III"]
                
                with open(min_csv_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    
                    col1 = entry.get("Column I", [])
                    col2 = entry.get("Column II", [])
                    col3 = entry.get("Column III", [])
                    
                    if isinstance(col1, str): col1 = [col1]
                    if isinstance(col2, str): col2 = [col2]
                    if isinstance(col3, str): col3 = [col3]
                    
                    max_len = max(len(col1), len(col2), len(col3))
                    
                    for i in range(max_len):
                        val1 = col1[i] if i < len(col1) else ""
                        val2 = col2[i] if i < len(col2) else ""
                        val3 = col3[i] if i < len(col3) else ""
                        writer.writerow([minister_name, i+1, val1, val2, val3])
                        
        except Exception as e:
             self.log(f"Error creating minister folders: {e}")
