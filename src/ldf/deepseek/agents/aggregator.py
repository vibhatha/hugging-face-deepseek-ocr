import os
import json
from typing import List, Dict, Any, Optional

from ldf.deepseek.agents.base import Agent

class AggregatorAgent(Agent):
    """
    Agent 3: Aggregator
    Role: Stitch pages based on 'CONTINUATION' logic.
    """
    def __init__(self):
        super().__init__("Aggregator")

    def process(self, processed_pages: List[Dict[str, Any]], output_dir: Optional[str] = None, pdf_name: str = "") -> List[Dict[str, Any]]:
        self.log("Aggregating pages...")
        
        consolidated_ministers = []
        current_minister = None
        debug_log = [] # detailed log of decisions
        
        for page in processed_pages:
            data = page.get('json_data', {})
            page_num = page['page_num']
            
            # Normalize data to list
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                if not isinstance(item, dict): continue
                
                raw_name = item.get("Minister") or ""
                # Use extracted name directly for the record
                minister_name = raw_name.strip()
                
                # Normalization for Comparison Logic (Internal Use Only)
                norm_name = minister_name.upper().replace("HON.", "").replace("DR.", "").replace("MR.", "").strip()
                # Remove common continuation markers for comparison
                norm_name_clean = norm_name.replace("(CONTD", "").replace("(CONTINUED", "").replace("CONTD", "").replace("CONTINUED", "").strip(" .():")
                
                # Check current minister for comparison
                current_norm = ""
                if current_minister:
                    curr_raw = current_minister.get("Minister", "")
                    current_norm = curr_raw.upper().replace("HON.", "").replace("DR.", "").strip()
                    current_norm = current_norm.replace("(CONTD", "").replace("(CONTINUED", "").replace("CONTD", "").replace("CONTINUED", "").strip(" .():")
                
                is_continuation_keyword = (
                    minister_name == "CONTINUATION_FROM_PREVIOUS" or 
                    "CONTINUATION" in minister_name.upper() or
                    "(CONTD" in minister_name.upper() or
                    "(CONTINUED" in minister_name.upper()
                )
                
                # MERGE IF: Explicit Keyword OR Name Matches Previous (ignoring suffixes)
                should_merge = False
                if is_continuation_keyword:
                    should_merge = True
                elif current_minister and norm_name_clean and (norm_name_clean == current_norm):
                    should_merge = True
                    self.log(f"Detected name match for merge: '{minister_name}' == '{curr_raw}'")

                decision = "UNKNOWN"
                
                if should_merge:
                    if current_minister:
                        decision = "MERGED"
                        self.log(f"Merging continuation on Page {page_num} to {current_minister.get('Minister')}")
                        # Merge Logic
                        for col in ["Column I", "Column II", "Column III"]:
                            if col in item:
                                # Ensure we have target list
                                if col not in current_minister: current_minister[col] = []
                                
                                # Get new values
                                new_vals = item[col]
                                if isinstance(new_vals, str): new_vals = [new_vals]
                                if not isinstance(new_vals, list): new_vals = []
                                
                                current_minister[col].extend(new_vals)
                    else:
                        decision = "ORPHANED_CONTINUATION"
                        self.log(f"Warning: Orphaned continuation on Page {page_num}")
                        
                else:
                    # New Minister
                    if minister_name:
                        decision = "NEW_MINISTER"
                        current_minister = item.copy() # Start new
                        current_minister['Minister'] = minister_name # Keep raw name
                        current_minister['_source_pages'] = [page_num]
                        
                        # Ensure columns become lists if they are strings
                        for col in ["Column I", "Column II", "Column III"]:
                            val = current_minister.get(col, [])
                            if isinstance(val, str): current_minister[col] = [val]
                            
                        consolidated_ministers.append(current_minister)
                    else:
                        decision = "SKIPPED_EMPTY_NAME"
                        
                debug_log.append({
                    "page": page_num,
                    "raw_minister": raw_name,
                    "norm_name": norm_name_clean,
                    "decision": decision,
                    "target_minister": current_minister.get("Minister") if current_minister else None
                })

        # Save Debug Artifact
        if output_dir and pdf_name:
             inter_dir = os.path.join(output_dir, "intermediate", pdf_name)
             os.makedirs(inter_dir, exist_ok=True)
             debug_path = os.path.join(inter_dir, "aggregator_debug_log.json")
             with open(debug_path, 'w', encoding='utf-8') as f:
                 json.dump(debug_log, f, indent=2, ensure_ascii=False)
             self.log(f"Saved Aggregator debug log to: {debug_path}")
                         
        return consolidated_ministers
