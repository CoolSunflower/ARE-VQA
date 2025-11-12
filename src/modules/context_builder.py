"""
Module 2: Context Builder
Generates visual context and retrieves external knowledge using LLM
"""
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.ollama_wrapper import OllamaVisionModel, OllamaTextModel
from prompts.prompt_templates import format_visual_context_prompt, format_knowledge_agent_prompt
from utils import setup_logging
from config import TEMPERATURE_CONFIG

logger = setup_logging(__name__)


class ContextBuilder:
    """
    Context Builder for VQA
    
    Gathers visual context from images and optionally retrieves
    external knowledge using an LLM as a knowledge agent.
    """
    
    def __init__(self):
        """Initialize context builder with vision and text models"""
        self.vision_model = OllamaVisionModel()
        self.text_model = OllamaTextModel()
        logger.info("Initialized Context Builder")
    
    def build(
        self,
        image_path: Path,
        question: str,
        triage_output: Dict[str, str],
        config: Dict[str, bool],
        use_cache: bool = True
    ) -> str:
        """
        Build context for answering a VQA question
        
        Args:
            image_path: Path to the image
            question: The question text
            triage_output: Output from triage router with 'complexity' and 'knowledge' keys
            config: Configuration dict with 'use_knowledge' flag
            use_cache: Whether to use cached responses
            
        Returns:
            Unified context string with visual and optional knowledge sections
        """
        logger.info(f"Building context for question: {question}")
        logger.debug(f"Triage: {triage_output}, Config: {config}")
        
        # Step 1: Extract visual context
        visual_context = self._extract_visual_context(
            image_path, question, use_cache
        )
        
        # Step 2: Retrieve external knowledge if needed
        external_knowledge = ""
        if (config.get("use_knowledge", False) and 
            triage_output.get("knowledge") == "KNOWLEDGE-BASED"):
            logger.info("Knowledge-based question detected - retrieving external knowledge")
            external_knowledge = self._retrieve_knowledge(
                question, visual_context, use_cache
            )
        
        # Step 3: Combine into unified context
        context = self._combine_context(visual_context, external_knowledge)
        
        logger.info(f"Context built ({len(context)} chars total)")
        return context
    
    def _extract_visual_context(
        self, 
        image_path: Path, 
        question: str,
        use_cache: bool = True
    ) -> str:
        """
        Extract visual context from image using vision model
        
        Args:
            image_path: Path to the image
            question: The question (for question-aware description)
            use_cache: Whether to use cached responses
            
        Returns:
            Visual context description
        """
        logger.debug(f"Extracting visual context from image: {image_path}")
        
        prompt = format_visual_context_prompt(question)
        
        visual_context = self.vision_model.generate(
            prompt=prompt,
            image_path=image_path,
            temperature=TEMPERATURE_CONFIG['executor'],
            use_cache=use_cache
        )
        
        logger.debug(f"Visual context extracted ({len(visual_context)} chars)")
        return visual_context.strip()
    
    def _retrieve_knowledge(
        self,
        question: str,
        visual_context: str,
        use_cache: bool = True
    ) -> str:
        """
        Retrieve external knowledge using LLM as knowledge agent
        
        Args:
            question: The question text
            visual_context: The visual context already extracted
            use_cache: Whether to use cached responses
            
        Returns:
            External knowledge string
        """
        logger.debug(f"Retrieving external knowledge for question: {question}")
        
        # Use LLM as knowledge agent
        prompt = format_knowledge_agent_prompt(question, visual_context)
        
        try:
            knowledge = self.text_model.generate(
                prompt=prompt,
                temperature=TEMPERATURE_CONFIG['query_gen'],
                use_cache=use_cache,
                max_tokens=250  # Limit to ~250 tokens to avoid massive responses
            )
            
            # Strip think tags from Qwen
            import re
            knowledge = re.sub(r'<think>.*?</think>', '', knowledge, flags=re.DOTALL).strip()
            
            logger.info(f"Retrieved external knowledge ({len(knowledge)} chars)")
            logger.debug(f"Knowledge: {knowledge[:200]}...")
            
            return knowledge.strip()
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return ""
    
    def _combine_context(self, visual_context: str, external_knowledge: str) -> str:
        """
        Combine visual context and external knowledge into unified context
        
        Args:
            visual_context: Visual description from image
            external_knowledge: External knowledge (if any)
            
        Returns:
            Combined context string
        """
        context_parts = [f"VISUAL CONTEXT:\n{visual_context}"]
        
        if external_knowledge:
            context_parts.append(f"\n\nEXTERNAL KNOWLEDGE:\n{external_knowledge}")
        
        return "\n".join(context_parts)


def test_context_builder():
    """Test the context builder with sample questions"""
    from data.dataset_loader import AOKVQADataset
    
    logger.info("="*60)
    logger.info("Testing Context Builder")
    logger.info("="*60)
    
    builder = ContextBuilder()
    dataset = AOKVQADataset()
    
    # Load a few samples
    samples = dataset.load_split('val')[:2]
    
    for i, sample in enumerate(samples, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Sample {i}/{len(samples)}")
        logger.info(f"{'='*60}")
        logger.info(f"Question: {sample.question}")
        
        image_path = dataset.get_image_path(sample.split, sample.image_id)
        
        # Test 1: VISUAL question without knowledge
        logger.info("\n--- Test 1: VISUAL + No Knowledge ---")
        triage1 = {"complexity": "ATOMIC", "knowledge": "VISUAL"}
        config1 = {"use_knowledge": False}
        context1 = builder.build(image_path, sample.question, triage1, config1, use_cache=False)
        logger.info(f"Context length: {len(context1)} chars")
        logger.info(f"Has external knowledge: {'EXTERNAL KNOWLEDGE' in context1}")
        
        # Test 2: KNOWLEDGE-BASED question with knowledge
        logger.info("\n--- Test 2: KNOWLEDGE-BASED + Knowledge Enabled ---")
        triage2 = {"complexity": "ATOMIC", "knowledge": "KNOWLEDGE-BASED"}
        config2 = {"use_knowledge": True}
        context2 = builder.build(image_path, sample.question, triage2, config2, use_cache=False)
        logger.info(f"Context length: {len(context2)} chars")
        logger.info(f"Has external knowledge: {'EXTERNAL KNOWLEDGE' in context2}")
        
        if 'EXTERNAL KNOWLEDGE' in context2:
            # Extract and show the knowledge part
            parts = context2.split('\n\nEXTERNAL KNOWLEDGE:\n')
            if len(parts) > 1:
                knowledge = parts[1]
                logger.info(f"Knowledge preview: {knowledge[:200]}...")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Context Builder Test Complete")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    test_context_builder()
