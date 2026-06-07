"""
Ollama API Adapter for local LLM inference.

This adapter provides a simplified interface to Ollama for the defense-in-depth pipeline,
enabling CPU-only inference with sequential model loading.
"""

import requests
import json
from typing import Optional, Dict, Any


class OllamaModelAdapter:
    """Adapter for Ollama local inference API."""
    
    def __init__(
        self,
        model_name: str,
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize Ollama adapter.
        
        Args:
            model_name: Name of the Ollama model (e.g., "llama3.2:3b")
            ollama_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None
    ) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return ""
    
    def get_classification_score(self, prompt: str) -> float:
        """
        Get binary classification score from yes/no response.
        
        The prompt should be formatted to elicit a "yes" or "no" answer.
        
        Args:
            prompt: Classification prompt
            
        Returns:
            Score in [0, 1] where 1.0 = "yes" (harmful), 0.0 = "no" (safe)
        """
        response = self.generate(prompt, max_tokens=50, temperature=0.0)
        response_lower = response.lower().strip()
        
        # Check first 50 characters for yes/no
        first_part = response_lower[:50]
        
        if "yes" in first_part and "no" not in first_part:
            return 1.0
        elif "no" in first_part and "yes" not in first_part:
            return 0.0
        else:
            # Uncertain - default to safe
            print(f"Warning: Ambiguous classification response: {response[:100]}")
            return 0.5
    
    def check_available(self) -> bool:
        """
        Check if Ollama is running and model is available.
        
        Returns:
            True if model is available
        """
        try:
            response = self.session.get(
                f"{self.ollama_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m.get("name") == self.model_name for m in models)
        except requests.exceptions.RequestException:
            return False
    
    def __repr__(self) -> str:
        return f"OllamaModelAdapter(model={self.model_name}, url={self.ollama_url})"


class OllamaClassifier:
    """Wrapper for using Ollama models as classifiers."""
    
    def __init__(
        self,
        model_name: str,
        template: str,
        threshold: float = 0.5,
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize classifier.
        
        Args:
            model_name: Ollama model name
            template: Prompt template with {QUERY} or {RESPONSE} placeholder
            threshold: Classification threshold
            ollama_url: Ollama API URL
        """
        self.adapter = OllamaModelAdapter(model_name, ollama_url)
        self.template = template
        self.threshold = threshold
    
    def classify(self, text: str) -> tuple[bool, float]:
        """
        Classify text as harmful or safe.
        
        Args:
            text: Text to classify
            
        Returns:
            (is_flagged, score) tuple
        """
        # Format prompt
        if "{QUERY}" in self.template:
            prompt = self.template.replace("{QUERY}", text)
        elif "{RESPONSE}" in self.template:
            prompt = self.template.replace("{RESPONSE}", text)
        else:
            prompt = self.template + "\n\n" + text
        
        # Get score
        score = self.adapter.get_classification_score(prompt)
        is_flagged = score >= self.threshold
        
        return is_flagged, score
    
    def __repr__(self) -> str:
        return f"OllamaClassifier(model={self.adapter.model_name}, threshold={self.threshold})"
