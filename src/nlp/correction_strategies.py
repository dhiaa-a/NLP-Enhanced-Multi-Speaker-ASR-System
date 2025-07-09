"""
Correction Strategies Module
Contains different strategies for correcting ASR errors.
"""

import re
import torch
from transformers import BertForMaskedLM, BertTokenizer
from typing import List, Dict, Optional, Tuple
import editdistance
from loguru import logger
import nltk
from nltk.corpus import wordnet
import json

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class CorrectionStrategies:
    """
    Collection of different correction strategies for ASR errors.
    Each strategy is optimized for different types of errors.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self._init_models()
        self._load_resources()
    
    def _init_models(self):
        """Initialize models for different strategies"""
        try:
            # BERT for masked language modeling
            self.bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(self.device)
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            self.bert_model = None
    
    def _load_resources(self):
        """Load linguistic resources"""
        # Common ASR confusions and corrections
        self.common_corrections = {
            'thre': 'three',
            'tw': 'two',
            'fo': 'four',
            'agr': 'agree',
            'budg': 'budget',
            'connec': 'connection',
            'mee': 'meet',
            'tomor': 'tomorrow',
            'yest': 'yesterday',
            'prob': 'probably',
            'def': 'definitely',
            'consid': 'consider',
            'deadl': 'deadline',
            'fri': 'friday',
            'he': 'hear',
            'b': 'bad',
            'exact': 'exactly',
            'tha': 'that'
        }
        
        # Phonetically similar words
        self.phonetic_groups = {
            'to': ['two', 'too'],
            'there': ['their', 'they\'re'],
            'your': ['you\'re'],
            'its': ['it\'s'],
            'where': ['wear', 'were'],
            'know': ['no'],
            'right': ['write'],
            'meet': ['meat'],
            'week': ['weak']
        }
        
        # Common word beginnings for completion
        self.word_prefixes = self._load_word_prefixes()
    
    def _load_word_prefixes(self) -> Dict[str, List[str]]:
        """Load common word prefixes for completion"""
        # In production, load from a comprehensive dictionary
        return {
            'con': ['connection', 'conference', 'consider', 'continue', 'contact'],
            'meet': ['meeting', 'meetings'],
            'tom': ['tomorrow', 'tomato'],
            'yes': ['yesterday', 'yes'],
            'prob': ['probably', 'problem', 'problematic'],
            'def': ['definitely', 'define', 'default'],
            'proj': ['project', 'projection', 'projector'],
            'doc': ['document', 'doctor', 'documentation'],
            'rep': ['report', 'reply', 'represent', 'repeat']
        }
    
    def pattern_based_correction(self, text: str) -> str:
        """
        Apply rule-based patterns to fix common ASR errors.
        Fast and effective for predictable error patterns.
        """
        if not text:
            return text
            
        corrected = text
        
        # Fix incomplete words with dashes
        corrected = re.sub(r'(\w{3,})-\s*(?=\s|$)', r'\1', corrected)
        corrected = re.sub(r'(?:^|\s)-(\w{3,})', r' \1', corrected)
        
        # Fix words with ### corruption
        def fix_corrupted(match):
            word = match.group(1)
            # Try to find in common corrections
            if word in self.common_corrections:
                return self.common_corrections[word]
            # Try prefix matching
            for prefix, completions in self.word_prefixes.items():
                if word.startswith(prefix) and len(completions) > 0:
                    return completions[0]  # Return most common
            return word
        
        corrected = re.sub(r'(\w+)##+', fix_corrupted, corrected)
        
        # Remove repeated words (common in overlapping speech)
        corrected = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', corrected)
        
        # Fix spacing issues
        corrected = re.sub(r'\s+', ' ', corrected)
        corrected = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', ' ', corrected)
        
        # Apply common corrections
        words = corrected.split()
        corrected_words = []
        for word in words:
            cleaned = re.sub(r'[^\w\s]', '', word.lower())
            if cleaned in self.common_corrections:
                corrected_words.append(self.common_corrections[cleaned])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words).strip()
    
    def masked_language_model_correction(self, text: str) -> str:
        """
        Use BERT masked language model to predict corrupted parts.
        Good for context-dependent corrections.
        """
        if not self.bert_model or not text:
            return text
        
        # Identify corrupted parts
        corrupted_pattern = re.compile(r'##+|\b\w{1,2}\b|(?:^|\s)-\w*|\w*-(?:\s|$)')
        
        # Create masked version
        masked_text = text
        mask_count = 0
        
        # Replace corruptions with [MASK]
        for match in corrupted_pattern.finditer(text):
            if mask_count < 5:  # Limit masks to avoid confusion
                masked_text = masked_text[:match.start()] + '[MASK]' + masked_text[match.end():]
                mask_count += 1
        
        if mask_count == 0:
            return text
        
        try:
            # Tokenize
            inputs = self.bert_tokenizer(masked_text, return_tensors="pt", 
                                       max_length=128, truncation=True, 
                                       padding=True).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = outputs.logits
            
            # Get predicted tokens for masks
            input_ids = inputs.input_ids[0]
            mask_token_id = self.bert_tokenizer.mask_token_id
            masked_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            
            # Replace masks with predictions
            tokens = self.bert_tokenizer.convert_ids_to_tokens(input_ids)
            
            for idx in masked_indices:
                predicted_id = predictions[0, idx].argmax(axis=-1)
                predicted_token = self.bert_tokenizer.convert_ids_to_tokens([predicted_id])[0]
                tokens[idx] = predicted_token
            
            # Convert back to text
            corrected = self.bert_tokenizer.convert_tokens_to_string(tokens)
            corrected = corrected.replace('[CLS]', '').replace('[SEP]', '').strip()
            
            return corrected
            
        except Exception as e:
            logger.warning(f"Masked LM correction failed: {e}")
            return text
    
    def phonetic_correction(self, text: str) -> str:
        """
        Correct based on phonetic similarity.
        Useful for sound-alike word confusions.
        """
        if not text:
            return text
            
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Clean word
            cleaned = re.sub(r'[#-]+', '', word).lower()
            
            # Skip very short words unless they're known confusions
            if len(cleaned) < 3 and cleaned not in self.phonetic_groups:
                if len(cleaned) > 0:
                    corrected_words.append(word)
                continue
            
            # Check phonetic groups
            if cleaned in self.phonetic_groups:
                # In production, use context to select best option
                # For now, keep original if it's a valid option
                corrected_words.append(word)
            else:
                # Check for partial matches in corrections
                best_match = None
                min_distance = float('inf')
                
                for correct_word in self.common_corrections.values():
                    dist = editdistance.eval(cleaned, correct_word)
                    if dist < min_distance and dist <= 2:
                        min_distance = dist
                        best_match = correct_word
                
                if best_match:
                    corrected_words.append(best_match)
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def contextual_correction(self, text: str, context: List[str]) -> str:
        """
        Use context from previous utterances to correct current text.
        Best for conversation continuity.
        """
        if not text or not context:
            return text
        
        # Extract important words from context
        context_words = set()
        for ctx in context:
            # Extract nouns and verbs (content words)
            words = ctx.lower().split()
            context_words.update([w for w in words if len(w) > 3])
        
        # Correct based on context
        words = text.split()
        corrected_words = []
        
        for word in words:
            cleaned = re.sub(r'[#-]+', '', word).lower()
            
            if len(cleaned) < 3:
                # Try to match with context words
                matches = [cw for cw in context_words if cw.startswith(cleaned)]
                if matches:
                    corrected_words.append(matches[0])
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def word_completion(self, partial_word: str, context: Optional[str] = None) -> str:
        """
        Complete partial words based on common patterns.
        """
        cleaned = re.sub(r'[#-]+', '', partial_word).lower()
        
        if len(cleaned) < 2:
            return partial_word
        
        # Check direct matches
        if cleaned in self.common_corrections:
            return self.common_corrections[cleaned]
        
        # Check prefix matches
        for prefix, completions in self.word_prefixes.items():
            if cleaned.startswith(prefix):
                # If we have context, try to select best match
                # For now, return most common
                return completions[0]
        
        # Try WordNet for completion
        try:
            synsets = wordnet.synsets(cleaned)
            if synsets:
                return synsets[0].lemmas()[0].name()
        except:
            pass
        
        return partial_word
    
    def aggressive_reconstruction(self, text: str, context: List[str]) -> str:
        """
        Aggressively reconstruct heavily corrupted text.
        Used when other methods fail.
        """
        # Extract any valid words
        words = text.split()
        valid_words = []
        
        for word in words:
            cleaned = re.sub(r'[^a-zA-Z]', '', word)
            if len(cleaned) >= 3:
                valid_words.append(cleaned)
        
        if not valid_words:
            return "Could not understand the speech"
        
        # If we have context, try to form coherent sentence
        if context and len(context) > 0:
            # Simple template-based reconstruction
            if len(valid_words) == 1:
                return f"I think you said something about {valid_words[0]}"
            elif len(valid_words) == 2:
                return f"{valid_words[0]} {valid_words[1]}"
            else:
                return ' '.join(valid_words)
        
        return ' '.join(valid_words)
    
    def apply_all_strategies(self, text: str, context: List[str] = None) -> Dict[str, str]:
        """
        Apply all correction strategies and return results.
        Useful for comparison and ensemble approaches.
        """
        results = {
            'original': text,
            'pattern_based': self.pattern_based_correction(text),
            'masked_lm': self.masked_language_model_correction(text),
            'phonetic': self.phonetic_correction(text)
        }
        
        if context:
            results['contextual'] = self.contextual_correction(text, context)
        
        # Add aggressive reconstruction if text is severely corrupted
        if text.count('#') > 3 or text.count('-') > 3:
            results['aggressive'] = self.aggressive_reconstruction(text, context or [])
        
        return results