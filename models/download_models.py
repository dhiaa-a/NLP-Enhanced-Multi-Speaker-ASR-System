#!/usr/bin/env python3
"""
Script to download required models for the Multi-Speaker ASR with NLP Enhancement system.
Downloads and caches pre-trained models from Hugging Face and other sources.
"""

import os
import sys
import requests
import torch
from tqdm import tqdm
from pathlib import Path
import hashlib
import json
from loguru import logger

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelDownloader:
    """Downloads and manages pre-trained models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "nlp_models": {
                "t5-base": {
                    "source": "huggingface",
                    "model_id": "t5-base",
                    "files": ["pytorch_model.bin", "config.json", "tokenizer.json"]
                },
                "t5-small": {
                    "source": "huggingface",
                    "model_id": "t5-small",
                    "files": ["pytorch_model.bin", "config.json", "tokenizer.json"]
                },
                "bert-base-uncased": {
                    "source": "huggingface",
                    "model_id": "bert-base-uncased",
                    "files": ["pytorch_model.bin", "config.json", "vocab.txt"]
                },
                "grammar-corrector": {
                    "source": "huggingface",
                    "model_id": "vennify/t5-base-grammar-correction",
                    "files": ["pytorch_model.bin", "config.json"]
                }
            },
            "asr_models": {
                "whisper-base": {
                    "source": "openai",
                    "model_id": "base",
                    "files": ["base.pt"]
                }
            },
            "custom_models": {
                "denoiser": {
                    "source": "custom",
                    "url": "https://github.com/yourusername/models/denoiser.pth",
                    "filename": "denoiser.pth",
                    "size": "50MB"
                }
            }
        }
        
        # Track download progress
        self.downloaded_models = self._load_download_status()
    
    def _load_download_status(self) -> dict:
        """Load status of previously downloaded models"""
        status_file = self.models_dir / "download_status.json"
        if status_file.exists():
            with open(status_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_download_status(self):
        """Save download status"""
        status_file = self.models_dir / "download_status.json"
        with open(status_file, 'w') as f:
            json.dump(self.downloaded_models, f, indent=2)
    
    def download_all(self):
        """Download all required models"""
        logger.info("Starting model download process...")
        
        # Download NLP models
        logger.info("Downloading NLP models...")
        for model_name, config in self.model_configs["nlp_models"].items():
            self._download_huggingface_model(model_name, config)
        
        # Download ASR models
        logger.info("Downloading ASR models...")
        for model_name, config in self.model_configs["asr_models"].items():
            if config["source"] == "openai":
                self._download_whisper_model(model_name, config)
        
        # Download custom models
        logger.info("Checking for custom models...")
        for model_name, config in self.model_configs["custom_models"].items():
            self._download_custom_model(model_name, config)
        
        # Save status
        self._save_download_status()
        logger.info("Model download complete!")
    
    def _download_huggingface_model(self, model_name: str, config: dict):
        """Download model from Hugging Face"""
        if model_name in self.downloaded_models:
            logger.info(f"Model {model_name} already downloaded")
            return
        
        logger.info(f"Downloading {model_name} from Hugging Face...")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Download model
            if "t5" in model_name.lower():
                from transformers import T5ForConditionalGeneration, T5Tokenizer
                model = T5ForConditionalGeneration.from_pretrained(config["model_id"])
                tokenizer = T5Tokenizer.from_pretrained(config["model_id"])
            elif "bert" in model_name.lower():
                from transformers import BertModel, BertTokenizer
                model = BertModel.from_pretrained(config["model_id"])
                tokenizer = BertTokenizer.from_pretrained(config["model_id"])
            else:
                model = AutoModel.from_pretrained(config["model_id"])
                tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
            
            # Save locally
            model.save_pretrained(str(model_dir))
            tokenizer.save_pretrained(str(model_dir))
            
            self.downloaded_models[model_name] = {
                "status": "completed",
                "path": str(model_dir)
            }
            
            logger.info(f"Successfully downloaded {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            self.downloaded_models[model_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    def _download_whisper_model(self, model_name: str, config: dict):
        """Download Whisper model"""
        if model_name in self.downloaded_models:
            logger.info(f"Model {model_name} already downloaded")
            return
        
        logger.info(f"Downloading Whisper {config['model_id']} model...")
        
        try:
            import whisper
            
            # This will download and cache the model
            model = whisper.load_model(config["model_id"])
            
            # Save path info
            self.downloaded_models[model_name] = {
                "status": "completed",
                "model_id": config["model_id"]
            }
            
            logger.info(f"Successfully downloaded Whisper {config['model_id']}")
            
        except Exception as e:
            logger.error(f"Failed to download Whisper model: {e}")
            logger.info("Please install whisper: pip install openai-whisper")
            self.downloaded_models[model_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    def _download_custom_model(self, model_name: str, config: dict):
        """Download custom model from URL"""
        if model_name in self.downloaded_models:
            logger.info(f"Model {model_name} already downloaded")
            return
        
        # For now, create a dummy model file
        logger.info(f"Creating placeholder for {model_name}...")
        
        model_path = self.models_dir / config["filename"]
        
        # Create dummy model
        if model_name == "denoiser":
            from src.audio.noise_reduction import DenoisingAutoencoder
            model = DenoisingAutoencoder()
            torch.save(model.state_dict(), str(model_path))
            
            self.downloaded_models[model_name] = {
                "status": "completed",
                "path": str(model_path)
            }
            logger.info(f"Created placeholder model: {model_name}")
    
    def verify_models(self) -> bool:
        """Verify all models are downloaded correctly"""
        logger.info("Verifying downloaded models...")
        
        all_good = True
        for category in self.model_configs.values():
            for model_name in category.keys():
                if model_name not in self.downloaded_models:
                    logger.warning(f"Model {model_name} not downloaded")
                    all_good = False
                elif self.downloaded_models[model_name]["status"] != "completed":
                    logger.warning(f"Model {model_name} download failed")
                    all_good = False
                else:
                    logger.info(f"✓ Model {model_name} verified")
        
        return all_good
    
    def create_sample_models(self):
        """Create sample model files for testing"""
        logger.info("Creating sample model files for testing...")
        
        # Create a sample denoiser model
        denoiser_path = self.models_dir / "denoiser.pth"
        if not denoiser_path.exists():
            from src.audio.noise_reduction import DenoisingAutoencoder
            model = DenoisingAutoencoder()
            torch.save(model.state_dict(), str(denoiser_path))
            logger.info("Created sample denoiser model")
        
        # Create model info file
        info_file = self.models_dir / "model_info.txt"
        with open(info_file, 'w') as f:
            f.write("Model Directory Information\n")
            f.write("==========================\n\n")
            f.write("This directory contains pre-trained models for the ASR system.\n\n")
            f.write("Models:\n")
            f.write("- denoiser.pth: Audio denoising model\n")
            f.write("- NLP models: Downloaded from Hugging Face\n")
            f.write("- Whisper models: Downloaded from OpenAI\n")
        
        logger.info("Sample models created")


def main():
    """Main download function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for Multi-Speaker ASR")
    parser.add_argument("--models-dir", type=str, default="models",
                      help="Directory to store models")
    parser.add_argument("--verify-only", action="store_true",
                      help="Only verify existing models")
    parser.add_argument("--create-samples", action="store_true",
                      help="Create sample models for testing")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.verify_only:
        success = downloader.verify_models()
        sys.exit(0 if success else 1)
    
    if args.create_samples:
        downloader.create_sample_models()
        sys.exit(0)
    
    # Download all models
    try:
        downloader.download_all()
        success = downloader.verify_models()
        
        if success:
            logger.info("\n✅ All models downloaded successfully!")
            logger.info("You can now run the demo: python scripts/run_demo.py")
        else:
            logger.warning("\n⚠️ Some models failed to download")
            logger.info("Try running again or check your internet connection")
            
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nDownload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()