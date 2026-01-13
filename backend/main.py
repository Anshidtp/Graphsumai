import logging
import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from config.settings import settings
from src.data_preprocessor.loader import DataLoader
from src.data_preprocessor.entity_resolver import EntityResolver
from src.data_preprocessor.transformer import DataProcessor
from src.data_preprocessor.graph_enricher import GraphEnricher
from src.vectorization.embedder import Embedder
from src.k_graph.graph_builder import GraphConstructor  
from src.k_graph.query_engine import QueryEngine as GraphQueryEngine
from src.rag.retriever import KnowledgeGraphRetriever
from src.rag.generator import RagGenerator as LlamaGenerator
from src.api.routes import router, set_dependencies

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Complete production pipeline with triplet embeddings"""
    
    def __init__(self):
        self.loader = None
        self.resolver = None
        self.processor = None
        self.enricher = None
        self.embedder = None
        self.constructor = None
        self.query_engine = None
        self.retriever = None
        self.generator = None
    
    def step1_resolve_entities(self):
        """
        STEP 1: Resolve all Freebase IDs to readable names
        Saves: 
        - entities_resolved.csv
        - triplets_readable.csv (NO IDs, only human-readable text)
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 1: ENTITY RESOLUTION WITH WIKIDATA")
        logger.info("="*70)
        
        # Load data
        self.loader = DataLoader(settings.DATA_DIR)
        self.loader.load_all()
        
        # Initialize resolver
        self.resolver = EntityResolver(str(settings.CACHE_FILE))
        self.processor = DataProcessor(self.resolver)
        
        # Resolve entities
        entities_file = settings.RESOLVED_DIR / "entities_resolved.csv"
        self.processor.resolve_and_save_entities(
            self.loader.entities,
            entities_file
        )
        
        # Resolve triplets (NO IDs in output!)
        triplets_file = settings.RESOLVED_DIR / "triplets_readable.csv"
        self.processor.resolve_and_save_triplets(
            self.loader.triples['train'],
            triplets_file
        )
        
        logger.info("\n✅ Step 1 Complete! Files saved:")
        logger.info(f"  - {entities_file}")
        logger.info(f"  - {triplets_file}")
        
        # Optional: Enrich with LLM-extracted knowledge
        if settings.LLM_API_KEY:
            logger.info("\n🔄 Step 1.5: ENRICHING GRAPH WITH LLM KNOWLEDGE")
            self.enricher = GraphEnricher(settings.LLM_API_KEY, settings.LLM_MODEL)
            
            # Load the resolved data
            entities_df = pd.read_csv(entities_file)
            triplets_df = pd.read_csv(triplets_file)
            
            # Enrich a sample of entities
            enriched_df = self.enricher.batch_enrich_entities(
                entities_df, triplets_df, sample_size=50
            )
            
            if not enriched_df.empty:
                # Save enriched triplets
                enriched_file = settings.RESOLVED_DIR / "triplets_enriched.csv"
                enriched_df.to_csv(enriched_file, index=False)
                logger.info(f"  - {enriched_file} ({len(enriched_df)} new triplets)")
                
                # Merge with original triplets for downstream processing
                combined_triplets = pd.concat([triplets_df, enriched_df], ignore_index=True)
                combined_file = settings.RESOLVED_DIR / "triplets_combined.csv"
                combined_triplets.to_csv(combined_file, index=False)
                logger.info(f"  - {combined_file} ({len(combined_triplets)} total triplets)")
    
    def step2_build_graph(self, clear_existing: bool = True):
        """
        STEP 2: Build knowledge graph with triplet embeddings
        
        Process:
        1. Load readable triplets
        2. Generate embeddings for each triplet sentence
        3. Create graph with entities, relationships, and triplet nodes
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: BUILDING KNOWLEDGE GRAPH WITH TRIPLET EMBEDDINGS")
        logger.info("="*70)
        
        # Load resolved triplets (use combined if available)
        combined_file = settings.RESOLVED_DIR / "triplets_combined.csv"
        triplets_file = combined_file if combined_file.exists() else settings.RESOLVED_DIR / "triplets_readable.csv"
        
        if not triplets_file.exists():
            logger.error("❌ triplets_readable.csv not found. Run step1 first!")
            return
        
        triplets_df = pd.read_csv(triplets_file)
        logger.info(f"📂 Loaded {len(triplets_df):,} readable triplets")
        
        # Filter out rows with missing head_name or tail_name
        triplets_df = triplets_df.dropna(subset=['head_name', 'tail_name'])
        triplets_df['head_name'] = triplets_df['head_name'].astype(str)
        triplets_df['tail_name'] = triplets_df['tail_name'].astype(str)
        logger.info(f"📂 After filtering: {len(triplets_df):,} valid triplets")
        self.embedder = Embedder(settings.EMBEDDING_MODEL, cache_dir=settings.EMBEDDING_CACHE_DIR)
        self.constructor = GraphConstructor(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD
        )
        
        # Clear and create schema
        if clear_existing:
            self.constructor.clear_database()
        self.constructor.create_schema()
        
        # Generate embeddings for triplet texts
        triplet_texts = triplets_df['triplet_text'].tolist()
        embeddings = self.embedder.batch_embed_triplets(triplet_texts)
        
        # Prepare graph data
        logger.info("\n📝 Preparing graph data...")
        graph_data = []
        
        for i, row in triplets_df.iterrows():
            # Sanitize relation for Cypher
            relation_type = row['relation'].upper().replace(' ', '_').replace('-', '_')
            
            graph_data.append({
                'triplet_text': row['triplet_text'],
                'head_name': row['head_name'],
                'tail_name': row['tail_name'],
                'relation': row['relation'],
                'relation_type': relation_type,
                'embedding': embeddings[i].tolist(),
                'head_aliases': row['head_name'].lower().split(),
                'tail_aliases': row['tail_name'].lower().split()
            })
        
        # Create graph
        self.constructor.batch_create_graph_from_triplets(graph_data, settings.BATCH_SIZE)
        
        # Get statistics
        stats = self.constructor.get_statistics()
        
        logger.info("\n📊 Graph Construction Complete!")
        logger.info(f"  Entities: {stats['entities']:,}")
        logger.info(f"  Triplets: {stats['triplets']:,}")
        logger.info(f"  Relationships: {stats['relationships']:,}")
        logger.info(f"\n  Sample entities: {', '.join(stats['sample_entities'])}")
        
        logger.info("\n✅ Step 2 Complete!")
    
    def step3_initialize_rag(self):
        """
        STEP 3: Initialize RAG system
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: INITIALIZING RAG SYSTEM")
        logger.info("="*70)
        
        if not self.embedder:
            self.embedder = Embedder(settings.EMBEDDING_MODEL)
        
        if not self.constructor:
            self.constructor = GraphConstructor(
                settings.NEO4J_URI,
                settings.NEO4J_USER,
                settings.NEO4J_PASSWORD
            )
        
        self.query_engine = GraphQueryEngine(self.constructor.driver)
        self.retriever = KnowledgeGraphRetriever(
            self.query_engine,
            self.embedder
        )
        self.generator = LlamaGenerator(
            model_name=settings.LLM_MODEL,
            llm_api_key=settings.LLM_API_KEY,
            retriever=self.retriever
        )
        
        logger.info("✅ RAG system initialized")
        logger.info("  - Query Engine: Ready")
        logger.info("  - Retriever: Ready")
        logger.info("  - Generator: Ready")
    
    def step4_test_query(self, query: str):
        """
        STEP 4: Test the system with a query
        """
        logger.info("\n" + "="*70)
        logger.info(f"TESTING QUERY: {query}")
        logger.info("="*70)
        
        if not self.retriever or not self.generator:
            self.step3_initialize_rag()
        
        # Retrieve
        retrieval_result = self.retriever.retrieve(query)
        
        # Generate
        generation_result = self.generator.generate(
            query,
            retrieval_result['context']
        )
        
        logger.info(f"\n💡 ANSWER:")
        logger.info(f"{generation_result['answer']}")
        logger.info(f"\n📊 METADATA:")
        logger.info(f"  Found triplets: {retrieval_result['metadata']['found']}")
        
        return generation_result
    
    def step5_run_api(self):
        """
        STEP 5: Run FastAPI server
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 5: STARTING API SERVER")
        logger.info("="*70)
        
        # Initialize RAG if not already done
        if not self.query_engine:
            self.step3_initialize_rag()
        
        # Set dependencies for routes
        set_dependencies(self.query_engine, self.retriever, self.generator)
        
        # Create FastAPI app
        app = FastAPI(
            title="Graph RAG API",
            description="Production Graph RAG with Embeddings",
            version="2.0.0"
        )
        
        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include router
        app.include_router(router, prefix="/api/v1", tags=["Graph RAG"])
        
        logger.info("\n🌐 API Server Configuration:")
        logger.info(f"  Host: {settings.API_HOST}")
        logger.info(f"  Port: {settings.API_PORT}")
        logger.info(f"  Docs: http://localhost:{settings.API_PORT}/docs")
        
        # Run server
        uvicorn.run(
            app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            log_level="info"
        )


def main():
    parser = argparse.ArgumentParser(
        description="FB15k-237 Complete Graph RAG System"
    )
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Execute specific step'
    )
    parser.add_argument('--query', type=str, help='Test query (for step 4)')
    parser.add_argument('--clear', action='store_true', help='Clear existing graph (step 2)')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    pipeline = GraphRAGPipeline()
    
    try:
        if args.all:
            # Run complete pipeline
            pipeline.step1_resolve_entities()
            pipeline.step2_build_graph(clear_existing=True)
            pipeline.step3_initialize_rag()
            pipeline.step4_test_query("What are some professions in the knowledge graph?")
            pipeline.step5_run_api()
        
        elif args.step == 1:
            pipeline.step1_resolve_entities()
        
        elif args.step == 2:
            pipeline.step2_build_graph(clear_existing=args.clear)
        
        elif args.step == 3:
            pipeline.step3_initialize_rag()
        
        elif args.step == 4:
            query = args.query or "What are some entities in the knowledge graph?"
            pipeline.step4_test_query(query)
        
        elif args.step == 5:
            pipeline.step5_run_api()
        
        else:
            parser.print_help()
    
    finally:
        if pipeline.constructor:
            pipeline.constructor.close()


if __name__ == "__main__":
    main()