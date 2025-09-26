#!/usr/bin/env python3
"""
Data collection script for trading card images and metadata.
Collects 50,000+ cards from multiple APIs with proper rate limiting and validation.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
import json
import yaml
from datetime import datetime
import aiohttp
import aiofiles

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.api_clients import DataCollectionOrchestrator
from data.validation import DataValidationPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataCollectionManager:
    """Manages the complete data collection process."""
    
    def __init__(self, config_path: str = "configs/data_collection.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.collection_targets = self.config['collection_targets']
        self.image_config = self.config['image_processing']
        
        # Initialize components
        pokemon_api_key = os.getenv('POKEMON_TCG_API_KEY')
        self.orchestrator = DataCollectionOrchestrator(pokemon_api_key)
        self.validator = DataValidationPipeline(config_path)
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for data storage."""
        base_path = Path(self.image_config['storage']['local_path'])
        
        directories = [
            base_path,
            base_path / "mtg",
            base_path / "pokemon", 
            base_path / "yugioh",
            Path("logs"),
            Path("data/processed"),
            Path("data/annotations")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    async def collect_card_metadata(self) -> Dict[str, List[Dict]]:
        """Collect card metadata from all APIs."""
        logger.info("Starting card metadata collection...")
        
        # Calculate collection targets
        total_target = self.collection_targets['phase_1_minimum']
        distribution = self.collection_targets['distribution']
        
        targets = {
            'mtg': int(total_target * distribution['mtg']),
            'pokemon': int(total_target * distribution['pokemon']),
            'yugioh': int(total_target * distribution['yugioh'])
        }
        
        logger.info(f"Collection targets: {targets}")
        
        # Collect from all APIs
        results = await self.orchestrator.collect_all_cards(
            limit_per_game=max(targets.values())
        )
        
        # Trim to target sizes
        for game, target_count in targets.items():
            if game in results and len(results[game]) > target_count:
                results[game] = results[game][:target_count]
        
        # Save metadata
        await self._save_metadata(results)
        
        return results
    
    async def _save_metadata(self, card_data: Dict[str, List[Dict]]):
        """Save collected metadata to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for game, cards in card_data.items():
            filename = f"data/processed/{game}_metadata_{timestamp}.json"
            
            async with aiofiles.open(filename, 'w') as f:
                await f.write(json.dumps(cards, indent=2, default=str))
            
            logger.info(f"Saved {len(cards)} {game} cards to {filename}")
    
    async def download_card_images(self, card_data: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Download card images from collected metadata."""
        logger.info("Starting card image downloads...")
        
        download_stats = {}
        
        for game, cards in card_data.items():
            logger.info(f"Downloading {game} card images...")
            
            game_stats = await self._download_game_images(game, cards)
            download_stats[game] = game_stats
            
            logger.info(f"Downloaded {game_stats['success']} {game} images, {game_stats['failed']} failed")
        
        return download_stats
    
    async def _download_game_images(self, game: str, cards: List[Dict]) -> Dict[str, int]:
        """Download images for a specific game."""
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Create game directory
        game_dir = Path(self.image_config['storage']['local_path']) / game
        game_dir.mkdir(exist_ok=True)
        
        # Semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(10)
        
        async def download_card_image(card: Dict) -> None:
            async with semaphore:
                try:
                    # Extract image URLs based on game type
                    image_urls = self._extract_image_urls(card, game)
                    
                    if not image_urls:
                        stats['skipped'] += 1
                        return
                    
                    # Use the first/best quality image
                    image_url = image_urls[0]
                    
                    # Generate filename
                    filename = self._generate_filename(card, game, image_url)
                    filepath = game_dir / filename
                    
                    # Skip if already exists
                    if filepath.exists():
                        stats['skipped'] += 1
                        return
                    
                    # Download image
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as response:
                            if response.status == 200:
                                content = await response.read()
                                
                                async with aiofiles.open(filepath, 'wb') as f:
                                    await f.write(content)
                                
                                stats['success'] += 1
                            else:
                                logger.warning(f"Failed to download {image_url}: HTTP {response.status}")
                                stats['failed'] += 1
                
                except Exception as e:
                    logger.error(f"Error downloading image for card: {e}")
                    stats['failed'] += 1
        
        # Download all images concurrently
        tasks = [download_card_image(card) for card in cards]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return stats
    
    def _extract_image_urls(self, card: Dict, game: str) -> List[str]:
        """Extract image URLs from card data based on game type."""
        urls = []
        
        if game == 'mtg':
            # Scryfall format
            if 'image_uris' in card:
                # Prefer high quality images
                for quality in ['large', 'normal', 'small']:
                    if quality in card['image_uris']:
                        urls.append(card['image_uris'][quality])
                        break
        
        elif game == 'pokemon':
            # Pokemon TCG API format
            if 'images' in card:
                # Prefer large images
                for quality in ['large', 'small']:
                    if quality in card['images']:
                        urls.append(card['images'][quality])
                        break
        
        elif game == 'yugioh':
            # YGOPRODeck format
            if 'card_images' in card and card['card_images']:
                for img_data in card['card_images']:
                    if 'image_url' in img_data:
                        urls.append(img_data['image_url'])
                        break
        
        return urls
    
    def _generate_filename(self, card: Dict, game: str, image_url: str) -> str:
        """Generate standardized filename for card image."""
        # Extract file extension
        ext = Path(image_url).suffix or '.jpg'
        
        # Get card identifiers
        if game == 'mtg':
            set_code = card.get('set', 'unknown')
            card_number = card.get('collector_number', 'unknown')
            card_id = card.get('id', 'unknown')[:8]  # First 8 chars of ID
        
        elif game == 'pokemon':
            set_code = card.get('set', {}).get('id', 'unknown')
            card_number = card.get('number', 'unknown')
            card_id = card.get('id', 'unknown')
        
        elif game == 'yugioh':
            set_code = 'ygo'  # YGO doesn't have clear set codes in API
            card_number = str(card.get('id', 'unknown'))
            card_id = card_number
        
        else:
            set_code = 'unknown'
            card_number = 'unknown'
            card_id = 'unknown'
        
        # Clean identifiers for filename
        set_code = self._clean_filename(set_code)
        card_number = self._clean_filename(card_number)
        card_id = self._clean_filename(card_id)
        
        return f"{game}_{set_code}_{card_number}_{card_id}{ext}"
    
    def _clean_filename(self, text: str) -> str:
        """Clean text for use in filename."""
        # Replace problematic characters
        cleaned = str(text).replace('/', '_').replace('\\', '_').replace(' ', '_')
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c in '_-.')
        return cleaned[:50]  # Limit length
    
    async def validate_collected_data(self) -> Dict:
        """Validate all collected data."""
        logger.info("Starting data validation...")
        
        data_dir = self.image_config['storage']['local_path']
        results = await self.validator.validate_dataset(data_dir)
        
        # Save validation report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"validation_report_{timestamp}.json"
        self.validator.save_validation_report(results, report_path)
        
        return results
    
    async def run_complete_collection(self) -> Dict:
        """Run the complete data collection pipeline."""
        logger.info("Starting complete data collection pipeline...")
        
        try:
            # Step 1: Collect metadata
            card_data = await self.collect_card_metadata()
            
            # Step 2: Download images
            download_stats = await self.download_card_images(card_data)
            
            # Step 3: Validate data
            validation_results = await self.validate_collected_data()
            
            # Compile final results
            results = {
                'collection_completed': datetime.now().isoformat(),
                'card_counts': {game: len(cards) for game, cards in card_data.items()},
                'download_stats': download_stats,
                'validation_results': validation_results,
                'api_stats': self.orchestrator.collection_stats
            }
            
            # Save final report
            with open('collection_report.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("Data collection pipeline completed successfully!")
            return results
        
        except Exception as e:
            logger.error(f"Data collection pipeline failed: {e}")
            raise


async def main():
    """Main entry point for data collection."""
    print("🃏 Trading Card Data Collection Pipeline")
    print("=" * 50)
    
    # Check environment
    if not os.path.exists('configs/data_collection.yaml'):
        print("❌ Configuration file not found: configs/data_collection.yaml")
        return
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    try:
        # Initialize and run collection
        manager = DataCollectionManager()
        results = await manager.run_complete_collection()
        
        # Print summary
        print("\n🎉 Collection Complete!")
        print(f"Cards collected: {sum(results['card_counts'].values())}")
        print(f"Images downloaded: {sum(stats['success'] for stats in results['download_stats'].values())}")
        print(f"Validation report: validation_report_*.json")
        print(f"Full report: collection_report.json")
        
    except KeyboardInterrupt:
        print("\n⏹️ Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        logger.exception("Collection failed with exception")


if __name__ == "__main__":
    asyncio.run(main())
