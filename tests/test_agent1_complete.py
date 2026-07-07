# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Test Agent 1 con GLM-4-9B y prompts contrastivos V2"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.base import BaseAgent
from agents.prompts import format_agent1_prompt_v2, TEMPERATURE_CONFIG
import json
import torch

class Agent1Protocol(BaseAgent):
    def __init__(self):
        super().__init__("agent1_protocol")
        
    def generate_protocol(self, user_query):
        """Genera protocolo PICO/SPIDER/PECO/PCC usando prompts contrastivos v2"""
        print(f"\n🔍 Query: '{user_query}'")
        
        # Cargar modelo si no está cargado
        if self.model is None:
            self.load_model()
        
        # Usar prompt optimizado v2
        prompt = format_agent1_prompt_v2(user_query)
        temp = TEMPERATURE_CONFIG['agent1_protocol']
        
        print(f"⚙️  Generando con GLM-4-9B (temp={temp})...")
        
        # Generar
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temp,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response[len(prompt):].strip()
        
        # Parsear JSON
        try:
            json_start = response_only.find('{')
            json_end = response_only.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_only[json_start:json_end]
                protocol = json.loads(json_str)
                
                framework = protocol.get('framework', 'UNKNOWN')
                print(f"✅ Framework detectado: {framework}")
                print(f"📋 Question: {protocol.get('research_question', 'N/A')[:80]}...")
                
                return protocol
            else:
                raise ValueError("No JSON encontrado")
                
        except Exception as e:
            print(f"⚠️  Error: {e}")
            print(f"Raw (500 chars): {response_only[:500]}")
            return {"error": str(e), "raw": response_only[:1000]}


# TESTS
if __name__ == "__main__":
    agent = Agent1Protocol()
    
    tests = [
        ("PICO", "¿Es efectiva la terapia cognitivo-conductual versus medicación para tratar depresión en adultos?"),
        ("SPIDER", "¿Qué barreras perciben los médicos de atención primaria para prescribir opioides en dolor crónico?"),
        ("PECO", "¿El consumo de alcohol durante el embarazo aumenta el riesgo de malformaciones congénitas?"),
    ]
    
    results = []
    
    for i, (expected, query) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(tests)}: {expected}")
        print(f"{'='*80}")
        
        result = agent.generate_protocol(query)
        detected = result.get('framework', 'ERROR')
        
        correct = detected == expected
        status = "✅ CORRECTO" if correct else f"❌ INCORRECTO (esperaba {expected})"
        results.append((expected, detected, correct))
        
        print(f"\n{status}")
        
        # Mostrar JSON parcial
        if 'error' not in result:
            print(f"\nJSON preview:")
            print(json.dumps(result, indent=2, ensure_ascii=False)[:600])
    
    # Resumen final
    print(f"\n\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")
    
    for i, (expected, detected, correct) in enumerate(results, 1):
        symbol = "✅" if correct else "❌"
        print(f"{symbol} Test {i}: Esperado {expected} | Detectado {detected}")
    
    accuracy = sum(1 for _, _, c in results if c) / len(results) * 100
    print(f"\n📊 Accuracy total: {accuracy:.0f}% ({sum(1 for _, _, c in results if c)}/{len(results)} correctos)")
    
    if accuracy == 100:
        print("\n🎉 ¡PERFECTO! Contrastive learning funciona correctamente")
    elif accuracy >= 66:
        print("\n👍 Bien, pero revisar casos fallidos")
    else:
        print("\n⚠️  Necesita ajuste de prompts")
