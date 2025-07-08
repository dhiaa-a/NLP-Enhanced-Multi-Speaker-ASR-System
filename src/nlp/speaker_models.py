"""
Speaker-Aware Models Module
Maintains speaker-specific language models and patterns for better correction.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional, Tuple
import re
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class SpeakerProfile:
    """Profile for an individual speaker"""
    speaker_id: int
    vocabulary: Set[str] = field(default_factory=set)
    phrase_patterns: List[str] = field(default_factory=list)
    word_frequencies: Counter = field(default_factory=Counter)
    speaking_style: Dict[str, float] = field(default_factory=dict)
    utterance_history: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    total_utterances: int = 0
    average_utterance_length: float = 0.0
    common_errors: Dict[str, str] = field(default_factory=dict)

class SpeakerAwareCorrector:
    """
    Maintains speaker-specific models for improved correction accuracy.
    Learns from each speaker's patterns over time.
    """
    
    def __init__(self, max_history_size: int = 100):
        self.speaker_profiles: Dict[int, SpeakerProfile] = {}
        self.max_history_size = max_history_size
        self.global_vocabulary = set()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.speaker_vectors = {}
        
        logger.info("Speaker-aware corrector initialized")
    
    def get_or_create_profile(self, speaker_id: int) -> SpeakerProfile:
        """Get existing profile or create new one"""
        if speaker_id not in self.speaker_profiles:
            self.speaker_profiles[speaker_id] = SpeakerProfile(speaker_id=speaker_id)
            logger.info(f"Created new profile for speaker {speaker_id}")
        return self.speaker_profiles[speaker_id]
    
    def update_speaker_model(self, speaker_id: int, corrected_text: str, raw_text: Optional[str] = None):
        """
        Update speaker model with new utterance.
        
        Args:
            speaker_id: Speaker identifier
            corrected_text: The corrected text
            raw_text: Original raw text (for learning error patterns)
        """
        profile = self.get_or_create_profile(speaker_id)
        
        # Update utterance history
        profile.utterance_history.append(corrected_text)
        if len(profile.utterance_history) > self.max_history_size:
            profile.utterance_history.pop(0)
        
        # Update vocabulary
        words = corrected_text.lower().split()
        profile.vocabulary.update(words)
        self.global_vocabulary.update(words)
        
        # Update word frequencies
        profile.word_frequencies.update(words)
        
        # Update statistics
        profile.total_utterances += 1
        profile.average_utterance_length = (
            (profile.average_utterance_length * (profile.total_utterances - 1) + len(words)) 
            / profile.total_utterances
        )
        
        # Learn error patterns if raw text provided
        if raw_text:
            self._learn_error_patterns(profile, raw_text, corrected_text)
        
        # Update phrase patterns
        self._update_phrase_patterns(profile)
        
        # Update speaking style
        self._analyze_speaking_style(profile)
        
        profile.last_updated = datetime.now()
        
        # Update TF-IDF vectors for similarity matching
        self._update_tfidf_vectors()
    
    def _learn_error_patterns(self, profile: SpeakerProfile, raw_text: str, corrected_text: str):
        """Learn common error patterns for this speaker"""
        raw_words = raw_text.lower().split()
        corrected_words = corrected_text.lower().split()
        
        # Simple alignment - in production, use dynamic programming
        for i, raw_word in enumerate(raw_words):
            if i < len(corrected_words):
                corrected_word = corrected_words[i]
                if raw_word != corrected_word and '#' in raw_word or '-' in raw_word:
                    # Store the error pattern
                    cleaned_raw = re.sub(r'[#-]+', '', raw_word)
                    if cleaned_raw:
                        profile.common_errors[cleaned_raw] = corrected_word
    
    def _update_phrase_patterns(self, profile: SpeakerProfile):
        """Extract common phrase patterns from utterance history"""
        if len(profile.utterance_history) < 5:
            return
        
        # Extract n-grams from recent utterances
        all_ngrams = []
        for utterance in profile.utterance_history[-20:]:
            words = utterance.lower().split()
            # Extract 2-4 word phrases
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    all_ngrams.append(ngram)
        
        # Find frequent phrases
        ngram_counts = Counter(all_ngrams)
        
        # Store phrases that appear at least twice
        profile.phrase_patterns = [
            phrase for phrase, count in ngram_counts.most_common(50)
            if count >= 2
        ]
    
    def _analyze_speaking_style(self, profile: SpeakerProfile):
        """Analyze speaking style characteristics"""
        if len(profile.utterance_history) < 10:
            return
        
        recent_utterances = profile.utterance_history[-20:]
        
        # Calculate style metrics
        style_metrics = {
            'formality': 0.0,
            'verbosity': 0.0,
            'technical_level': 0.0,
            'question_frequency': 0.0
        }
        
        # Formality indicators
        formal_words = {'therefore', 'however', 'furthermore', 'nevertheless', 'regarding'}
        informal_words = {'yeah', 'gonna', 'wanna', 'stuff', 'things'}
        
        # Technical indicators
        technical_words = {'system', 'process', 'implement', 'framework', 'algorithm', 
                          'database', 'analysis', 'optimize', 'configuration'}
        
        total_words = 0
        formal_count = 0
        informal_count = 0
        technical_count = 0
        question_count = 0
        
        for utterance in recent_utterances:
            words = set(utterance.lower().split())
            total_words += len(words)
            
            formal_count += len(words.intersection(formal_words))
            informal_count += len(words.intersection(informal_words))
            technical_count += len(words.intersection(technical_words))
            
            if '?' in utterance:
                question_count += 1
        
        # Calculate metrics
        if total_words > 0:
            style_metrics['formality'] = (formal_count - informal_count) / total_words
            style_metrics['technical_level'] = technical_count / total_words
            style_metrics['verbosity'] = profile.average_utterance_length / 10.0  # Normalized
            style_metrics['question_frequency'] = question_count / len(recent_utterances)
        
        profile.speaking_style = style_metrics
    
    def _update_tfidf_vectors(self):
        """Update TF-IDF vectors for all speakers"""
        if len(self.speaker_profiles) < 2:
            return
        
        # Combine all speaker histories
        all_documents = []
        speaker_ids = []
        
        for speaker_id, profile in self.speaker_profiles.items():
            if profile.utterance_history:
                # Combine recent utterances into one document
                document = ' '.join(profile.utterance_history[-20:])
                all_documents.append(document)
                speaker_ids.append(speaker_id)
        
        if len(all_documents) > 1:
            try:
                # Fit TF-IDF
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_documents)
                
                # Store vectors
                for i, speaker_id in enumerate(speaker_ids):
                    self.speaker_vectors[speaker_id] = tfidf_matrix[i]
                    
            except Exception as e:
                logger.warning(f"Failed to update TF-IDF vectors: {e}")
    
    def speaker_specific_correction(self, text: str, speaker_id: int, context: List[str] = None) -> str:
        """
        Apply speaker-specific correction based on learned patterns.
        
        Args:
            text: Raw text to correct
            speaker_id: Speaker identifier
            context: Recent context
            
        Returns:
            Corrected text
        """
        profile = self.get_or_create_profile(speaker_id)
        
        # If new speaker with no history, return original
        if profile.total_utterances < 5:
            return text
        
        corrected = text
        
        # Apply learned error corrections
        corrected = self._apply_error_corrections(corrected, profile)
        
        # Apply vocabulary matching
        corrected = self._apply_vocabulary_matching(corrected, profile)
        
        # Apply phrase pattern matching
        corrected = self._apply_phrase_patterns(corrected, profile)
        
        # Style-based corrections
        corrected = self._apply_style_corrections(corrected, profile)
        
        return corrected
    
    def _apply_error_corrections(self, text: str, profile: SpeakerProfile) -> str:
        """Apply learned error pattern corrections"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            cleaned = re.sub(r'[#-]+', '', word.lower())
            
            # Check if this error pattern has been seen before
            if cleaned in profile.common_errors:
                corrected_words.append(profile.common_errors[cleaned])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _apply_vocabulary_matching(self, text: str, profile: SpeakerProfile) -> str:
        """Match partial words to speaker's vocabulary"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            if '#' in word or '-' in word:
                # Extract partial
                partial = re.sub(r'[#-]+', '', word.lower())
                
                if len(partial) >= 3:
                    # Find best match in speaker's vocabulary
                    matches = [
                        vocab_word for vocab_word in profile.vocabulary
                        if vocab_word.startswith(partial)
                    ]
                    
                    if matches:
                        # Prefer frequently used words
                        best_match = max(matches, key=lambda w: profile.word_frequencies[w])
                        corrected_words.append(best_match)
                        continue
            
            corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _apply_phrase_patterns(self, text: str, profile: SpeakerProfile) -> str:
        """Apply known phrase patterns"""
        if not profile.phrase_patterns:
            return text
        
        # Sort patterns by length (longer first)
        patterns = sorted(profile.phrase_patterns, key=len, reverse=True)
        
        corrected = text.lower()
        
        for pattern in patterns[:20]:  # Limit to top 20 patterns
            pattern_words = pattern.split()
            
            # Create regex to match partial pattern
            regex_parts = []
            for word in pattern_words:
                # Allow for corrupted versions
                regex_parts.append(f"({word}|{word[:3]}[#-]*)")
            
            pattern_regex = r'\b' + r'\s+'.join(regex_parts) + r'\b'
            
            try:
                if re.search(pattern_regex, corrected):
                    corrected = re.sub(pattern_regex, pattern, corrected)
            except:
                continue
        
        return corrected
    
    def _apply_style_corrections(self, text: str, profile: SpeakerProfile) -> str:
        """Apply corrections based on speaking style"""
        # If speaker is formal, correct informal contractions
        if profile.speaking_style.get('formality', 0) > 0.5:
            corrections = {
                "don't": "do not",
                "won't": "will not",
                "can't": "cannot",
                "wouldn't": "would not"
            }
            for informal, formal in corrections.items():
                text = text.replace(informal, formal)
        
        return text
    
    def find_similar_speaker(self, speaker_id: int) -> Optional[int]:
        """Find most similar speaker based on speaking patterns"""
        if speaker_id not in self.speaker_vectors or len(self.speaker_vectors) < 2:
            return None
        
        current_vector = self.speaker_vectors[speaker_id]
        max_similarity = -1
        most_similar = None
        
        for other_id, other_vector in self.speaker_vectors.items():
            if other_id != speaker_id:
                similarity = cosine_similarity(
                    current_vector.reshape(1, -1),
                    other_vector.reshape(1, -1)
                )[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = other_id
        
        return most_similar if max_similarity > 0.7 else None
    
    def get_speaker_statistics(self, speaker_id: int) -> Dict:
        """Get statistics for a speaker"""
        profile = self.speaker_profiles.get(speaker_id)
        if not profile:
            return {}
        
        return {
            'total_utterances': profile.total_utterances,
            'vocabulary_size': len(profile.vocabulary),
            'average_utterance_length': profile.average_utterance_length,
            'speaking_style': profile.speaking_style,
            'common_phrases': profile.phrase_patterns[:10],
            'most_frequent_words': profile.word_frequencies.most_common(10)
        }
    
    def save_profiles(self, filepath: str):
        """Save speaker profiles to disk"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.speaker_profiles, f)
            logger.info(f"Saved {len(self.speaker_profiles)} speaker profiles")
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")
    
    def load_profiles(self, filepath: str):
        """Load speaker profiles from disk"""
        try:
            with open(filepath, 'rb') as f:
                self.speaker_profiles = pickle.load(f)
            logger.info(f"Loaded {len(self.speaker_profiles)} speaker profiles")
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")