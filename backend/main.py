import logging
import argparse
from pathlib import Path

# Import all components
from config.settings import settings
from src.data_preprocessor.loader import DataLoader
from src.data_preprocessor.entity_resolver import EntityResolver
from src.data_preprocessor.transformer import DataPreprocessor
from src.k_graph.graph_builder import GraphConstructor
from src.k_graph.query_engine import QueryEngine
from src.rag.retriever import Retriever
from src.rag.generator import RAGGenerator
# from backend.src.evaluation.link_prediction import LinkPredictionEvaluator
# from backend.src.evaluation.rag_evaluator import RAGEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.FileHandler(settings.LOGS_DIR / 'pipeline.log'),
    #     logging.StreamHandler()
    # ]
)
logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Complete pipeline for FB15k-237 Graph RAG"""
    
    def __init__(self):
        self.loader = None
        self.resolver = None
        self.preprocessor = None
        self.graph_constructor = None
        self.query_engine = None
        self.retriever = None
        self.rag_generator = None
        
    def step1_load_data(self):
        """
        Step 1: Load FB15k-237 Dataset
        
        - Loads train.txt, valid.txt, test.txt
        - Parses tab-separated triples
        - Collects unique entities and relations
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Loading FB15k-237 Dataset")
        logger.info("=" * 60)
        
        self.loader = DataLoader(settings.DATA_DIR)
        triples = self.loader.load_all_splits()
        
        # Show statistics
        entity_stats = self.loader.get_entity_statistics()
        relation_stats = self.loader.get_relation_statistics()
        
        logger.info(f"\nEntity Statistics:")
        logger.info(f"  Total entities: {entity_stats['total_entities']}")
        logger.info(f"  Total relations: {entity_stats['total_relations']}")
        
        logger.info(f"\nTop 5 Relations:")
        for rel, count in relation_stats['most_common_relations'][:5]:
            logger.info(f"  {rel}: {count}")
        
        return triples
    
    def step2_initialize_components(self):
        """
        Step 2: Initialize Components
        
        - Entity resolver (for readable names)
        - Preprocessor (clean and format data)
        - Graph constructor (Neo4j connection)
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Initializing Components")
        logger.info("=" * 60)
        
        # Initialize entity resolver
        self.resolver = EntityResolver(settings.ENTITY_NAMES_FILE)
        logger.info("✅ Entity resolver initialized")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(self.resolver)
        logger.info("✅ Preprocessor initialized")
        
        # Initialize graph constructor
        self.graph_constructor = GraphConstructor(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD
        )
        logger.info("✅ Graph constructor initialized")
    
    def step3_build_graph(self, use_split: str = 'train'):
        """
        Step 3: Build Knowledge Graph in Neo4j
        
        - Creates entity nodes
        - Creates relationships from triples
        - Builds indexes for fast querying
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"STEP 3: Building Knowledge Graph ({use_split} split)")
        logger.info("=" * 60)
        
        triples = self.loader.triples[use_split]
        entities = self.loader.entities
        
        stats = self.graph_constructor.build_graph(
            triples, 
            entities, 
            self.preprocessor,
            clear=True
        )
        
        logger.info(f"\nGraph built successfully:")
        logger.info(f"  Entities: {stats['entity_count']}")
        logger.info(f"  Relationships: {stats['relationship_count']}")
        
        return stats
    
    def step4_initialize_rag(self):
        """
        Step 4: Initialize RAG System
        
        - Query engine for graph traversal
        - Retriever for context extraction
        - Generator for answer synthesis
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Initializing RAG System")
        logger.info("=" * 60)
        
        # Initialize query engine
        self.query_engine = QueryEngine(self.graph_constructor.driver)
        logger.info("✅ Query engine initialized")
        
        # Initialize retriever
        self.retriever = Retriever(
            self.query_engine,
            settings.EMBEDDING_MODEL
        )
        logger.info("✅ Retriever initialized")
        
        # Initialize RAG generator
        self.rag_generator = RAGGenerator(
            self.retriever,
            settings.LLM_MODEL
        )
        logger.info("✅ RAG generator initialized")
    
    def step5_test_queries(self):
        """
        Step 5: Test with Sample Queries
        
        - Run example questions
        - Show retrieval and generation process
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Testing Sample Queries")
        logger.info("=" * 60)
        
        # Get some sample entities
        sample_triples = self.loader.triples['train'][:5]
        
        test_questions = []
        for head, relation, tail in sample_triples:
            head_name = self.resolver.resolve_entity(head)
            rel_name = self.resolver.resolve_relation(relation)
            
            question = f"What is the {rel_name} of {head_name}?"
            test_questions.append(question)
        
        # Test queries
        for question in test_questions:
            logger.info(f"\n❓ Question: {question}")
            result = self.rag_generator.generate(question)
            logger.info(f"💡 Answer: {result['answer']}")
            logger.info("-" * 60)
    
    # def step6_evaluate_link_prediction(self):
    #     """
    #     Step 6: Evaluate Link Prediction
        
    #     - Standard KB completion metrics
    #     - MRR, Hits@1, Hits@3, Hits@10
    #     """
    #     logger.info("\n" + "=" * 60)
    #     logger.info("STEP 6: Evaluating Link Prediction")
    #     logger.info("=" * 60)
        
    #     evaluator = LinkPredictionEvaluator(self.query_engine)
        
    #     test_triples = self.loader.triples['test']
    #     entities = self.loader.entities
        
    #     metrics = evaluator.evaluate_test_set(
    #         test_triples,
    #         entities,
    #         k_values=[1, 3, 10]
    #     )
        
    #     logger.info(f"\nLink Prediction Metrics:")
    #     for metric, value in metrics.items():
    #         logger.info(f"  {metric}: {value:.4f}")
        
    #     return metrics
    
    # def step7_evaluate_rag(self):
    #     """
    #     Step 7: Evaluate RAG System
        
    #     - Generate QA pairs from triples
    #     - Evaluate answer accuracy
    #     """
    #     logger.info("\n" + "=" * 60)
    #     logger.info("STEP 7: Evaluating RAG System")
    #     logger.info("=" * 60)
        
    #     rag_evaluator = RAGEvaluator(self.rag_generator)
        
    #     # Create QA pairs from validation set
    #     qa_pairs = rag_evaluator.create_qa_from_triples(
    #         self.loader.triples['valid'],
    #         self.resolver,
    #         n_samples=20
    #     )
        
    #     metrics = rag_evaluator.evaluate(qa_pairs)
        
    #     logger.info(f"\nRAG Evaluation Metrics:")
    #     logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
    #     logger.info(f"  Correct: {metrics['correct']}/{metrics['total']}")
        
    #     # Save results
    #     rag_evaluator.save_results(settings.RESULTS_DIR / 'rag_results.csv')
        
    #     return metrics
    
    def run_full_pipeline(self):
        """Run complete end-to-end pipeline"""
        logger.info("\n" + "🚀" * 30)
        logger.info("Starting FB15k-237 Graph RAG Pipeline")
        logger.info("🚀" * 30 + "\n")
        
        try:
            # Execute all steps
            self.step1_load_data()
            self.step2_initialize_components()
            self.step3_build_graph()
            self.step4_initialize_rag()
            self.step5_test_queries()
            
            # Evaluation (optional, can be slow)
            # self.step6_evaluate_link_prediction()
            # self.step7_evaluate_rag()
            
            logger.info("\n" + "✅" * 30)
            logger.info("Pipeline completed successfully!")
            logger.info("✅" * 30)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        
        finally:
            if self.graph_constructor:
                self.graph_constructor.close()
    
    def interactive_query_mode(self):
        """Interactive mode for querying the graph"""
        logger.info("\n🤖 Interactive Query Mode (type 'exit' to quit)")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not question:
                    continue
                
                result = self.rag_generator.generate(question)
                print(f"\nAnswer: {result['answer']}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description='Graph RAG System')
    parser.add_argument('--mode', 
                       choices=['full', 'build', 'query', 'evaluate'],
                       default='full',
                       help='Execution mode')
    parser.add_argument('--split',
                       choices=['train', 'valid', 'test'],
                       default='train',
                       help='Dataset split to use for graph building')
    
    args = parser.parse_args()
    
    pipeline = GraphRAGPipeline()
    
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    
    elif args.mode == 'build':
        pipeline.step1_load_data()
        pipeline.step2_initialize_components()
        pipeline.step3_build_graph(use_split=args.split)
    
    elif args.mode == 'query':
        pipeline.step1_load_data()
        pipeline.step2_initialize_components()
        pipeline.step4_initialize_rag()
        pipeline.interactive_query_mode()
    
    elif args.mode == 'evaluate':
        pipeline.step1_load_data()
        pipeline.step2_initialize_components()
        pipeline.step4_initialize_rag()
        # pipeline.step6_evaluate_link_prediction()  # Uncomment for full eval
        # pipeline.step7_evaluate_rag()


if __name__ == "__main__":
    main()