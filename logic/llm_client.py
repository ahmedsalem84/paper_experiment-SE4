import requests
import json
import re

class LLMClient:
    # Use 127.0.0.1 for better network reliability with Ollama
    def __init__(self, model_name="llama2", base_url="http://127.0.0.1:11434"):
        self.model = model_name
        self.base_url = base_url
        self.cache = {} 
        self.decision_log = [] 

    def set_model(self, model_name):
        self.model = model_name

    def resolve_ambiguity(self, class_name, class_code, context_a, context_b, global_rag_context=""):
        cache_key = f"{class_name}|{context_a}|{context_b}"
        if cache_key in self.cache:
            return self.cache[cache_key]['decision']

        # Construct Prompt for Llama 2/3 reliability
        prompt = f"""
        ### System Context:
        Project Description: {global_rag_context[:800]} 

        ### Task:
        You are a Software Architect. Decide if the Class '{class_name}' belongs to Service A or Service B.
        
        ### Code Snippet:
        {class_code[:300]}

        ### Options:
        Service A contains these classes: {context_a}
        Service B contains these classes: {context_b}

        ### Instructions:
        Respond ONLY with a valid JSON object. Do not include any text outside the JSON.
        Example format: {{"decision": "A", "reason": "it handles customer data"}}

        ### Decision:
        """
        
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model, 
                "prompt": prompt, 
                "stream": False, 
                "options": {"temperature": 0.1}
            }
            
            res = requests.post(url, json=payload, timeout=60)
            
            if res.status_code == 200:
                raw_response = res.json().get('response', '')
                
                # --- Robust JSON Extraction ---
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        decision = str(data.get("decision", "A")).strip().upper()
                        decision = 'A' if 'A' in decision else 'B'
                        reason = data.get("reason", "Expert semantic alignment.")
                    except:
                        decision, reason = self._fallback_parsing(raw_response)
                else:
                    decision, reason = self._fallback_parsing(raw_response)

                result = {'decision': decision, 'reason': reason}
                self.cache[cache_key] = result
                
                self.decision_log.append({
                    "Class": class_name, "Service A": context_a, "Service B": context_b,
                    "Decision": decision, "Reason": reason, "Model": self.model
                })
                
                return decision
            else:
                print(f"ERROR: Ollama returned status {res.status_code}. Model {self.model} not found.")
        
        except requests.exceptions.ConnectionError:
             print("ERROR: Connection Refused. Check if Ollama is running.")
        except Exception as e:
            print(f"LLM Fail: {e}")
        
        return 'A' # Universal fallback

    def _fallback_parsing(self, text):
        text_upper = text.upper()
        if "SERVICE B" in text_upper or '"DECISION": "B"' in text_upper:
            return "B", "Inferred from raw text output due to JSON parsing failure."
        return "A", "Default fallback (Could not parse output)."