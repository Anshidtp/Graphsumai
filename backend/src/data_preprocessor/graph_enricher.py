from LLMGraphTransformer import LLMGraphTransformer
from LLMGraphTransformer.schema import NodeSchema, RelationshipSchema
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)

node_schemas = [
    NodeSchema("Person", ["name", "birth_year", "death_year", "nationalitie", "profession"], "Represents an individual"),
    NodeSchema("Organization", ["name", "founding_year", "industrie"], "Represents a group, company, or institution"),
    NodeSchema("Location", ["name"], "Represents a geographical area such as a city, country, or region"),
    NodeSchema("Award", ["name", "field"], "Represents an honor, prize, or recognition"),
    NodeSchema("Event", 
               ["name", "event_type", "start_date", "end_date"], 
               "Represents a significant occurrence"),

    NodeSchema("TimePeriod", 
               ["name", "start_year", "end_year"], 
               "Represents a historical or logical time range"),
    NodeSchema("Work", 
               ["title", "work_type", "publication_year"], 
               "Represents books, films, research papers, or creative works"),

    NodeSchema("Document", 
               ["title", "document_type", "created_date"], 
               "Represents structured or unstructured documents"),

    NodeSchema("Concept", 
               ["name", "domain", "description"], 
               "Represents an abstract idea or notion"),

    NodeSchema("Topic", 
               ["name", "category"], 
               "Represents a subject or theme"),
    NodeSchema("Product", 
               ["name", "category", "launch_year"], 
               "Represents a commercial or digital product"),

    NodeSchema("Service", 
               ["name", "service_type"], 
               "Represents a service offering"),

    NodeSchema("Market", 
               ["name", "region"], 
               "Represents a commercial or economic market"),
    NodeSchema("Technology", 
               ["name", "technology_type", "release_year"], 
               "Represents tools, frameworks, or platforms"),

    NodeSchema("Dataset", 
               ["name", "dataset_type", "source"], 
               "Represents structured or unstructured datasets"),

    NodeSchema("Model", 
               ["name", "model_type", "framework"], 
               "Represents ML or AI models"),

    NodeSchema("Algorithm", 
               ["name", "algorithm_type"], 
               "Represents computational methods or techniques"),
    NodeSchema("Policy", 
               ["name", "policy_type", "effective_year"], 
               "Represents rules or governance frameworks"),

    NodeSchema("Law", 
               ["name", "jurisdiction", "enacted_year"], 
               "Represents legal regulations"),
    NodeSchema("FinancialMetric", 
               ["name", "metric_type", "unit"], 
               "Represents financial or performance indicators"),

    NodeSchema("Transaction", 
               ["transaction_type", "amount", "currency", "date"], 
               "Represents financial or data transactions"),
    NodeSchema("Community", 
               ["name", "community_type"], 
               "Represents social or interest-based groups"),

    NodeSchema("DemographicGroup", 
               ["name", "criteria"], 
               "Represents a population segment"),
]

relationship_schemas = [
    RelationshipSchema("Person", "SPOUSE_OF", "Person"),
    RelationshipSchema("Person", "MEMBER_OF", "Organization", ["start_year", "end_year", "year"]),
    RelationshipSchema("Person", "AWARDED", "Award", ["year"]),
    RelationshipSchema("Person", "LOCATED_IN", "Location"),
    RelationshipSchema("Organization", "LOCATED_IN", "Location"),
    RelationshipSchema("Organization", "PARTNERED_WITH", "Organization", ["start_year", "end_year", "year"]),
    RelationshipSchema("Organization", "HOSTED", "Event", ["year"]),
    RelationshipSchema("Person", "MEMBER_OF", "Organization", ["role", "start_year", "end_year"]),
    RelationshipSchema("Person", "FOUNDER_OF", "Organization", ["year"]),
    RelationshipSchema("Person", "EMPLOYED_BY", "Organization", ["role", "start_year", "end_year"]),
    RelationshipSchema("Person", "BOARD_MEMBER_OF", "Organization", ["start_year", "end_year"]),
    RelationshipSchema("Person", "AWARDED", "Award", ["year"]),
    RelationshipSchema("Organization", "RECEIVED_AWARD", "Award", ["year"]),
    RelationshipSchema("Person", "BORN_IN", "Location"),
    RelationshipSchema("Person", "DIED_IN", "Location"),
    RelationshipSchema("Person", "LOCATED_IN", "Location"),

    RelationshipSchema("Organization", "LOCATED_IN", "Location"),
    RelationshipSchema("Event", "HELD_IN", "Location"),
    RelationshipSchema("Person", "PARTICIPATED_IN", "Event", ["role"]),
    RelationshipSchema("Organization", "ORGANIZED", "Event"),
    RelationshipSchema("Event", "OCCURRED_DURING", "TimePeriod"),
    RelationshipSchema("Person", "CREATED", "Work", ["year"]),
    RelationshipSchema("Organization", "PUBLISHED", "Work", ["year"]),
    RelationshipSchema("Work", "ABOUT", "Topic"),
    RelationshipSchema("Work", "MENTIONS", "Concept"),
    RelationshipSchema("Dataset", "USED_BY", "Model"),
    RelationshipSchema("Model", "BASED_ON", "Algorithm"),
    RelationshipSchema("Technology", "IMPLEMENTED_USING", "Technology"),
    RelationshipSchema("Model", "TRAINED_ON", "Dataset"),
    RelationshipSchema("Organization", "PRODUCES", "Product"),
    RelationshipSchema("Organization", "PROVIDES", "Service"),
    RelationshipSchema("Product", "TARGETS", "Market"),
    RelationshipSchema("Event", "CAUSED_BY", "Event"),
    RelationshipSchema("Event", "RESULTED_IN", "Event"),
    RelationshipSchema("Event", "INVOLVES", "Concept"),
    RelationshipSchema("Concept", "RELATED_TO", "Concept"),
    RelationshipSchema("Topic", "SUBTOPIC_OF", "Topic"),
    RelationshipSchema("Concept", "BELONGS_TO", "Domain")
]

class GraphEnricher:
    """Use LLM to extract additional knowledge from existing entities"""

    def __init__(self, api_key: str, model_name: str = "llama3-70b-8192"):
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.1,
            api_key=api_key
        )
        self.graph_transformer = LLMGraphTransformer(llm=self.llm,
                                                     allowed_nodes=node_schemas,
                                                     allowed_relationships=relationship_schemas)

    def enrich_entity(self, entity_name: str, existing_triplets: List[str]) -> List[dict]:
        """
        Extract additional knowledge about an entity using LLM

        Args:
            entity_name: Name of entity to enrich
            existing_triplets: Existing triplets about this entity

        Returns:
            List of new triplets as dicts
        """
        # Create context document
        context = f"Entity: {entity_name}\n"
        context += "Known facts:\n" + "\n".join(existing_triplets[:5])  # Limit context

        document = Document(
            page_content=f"""
            Extract additional factual knowledge about {entity_name}.
            Focus on relationships, attributes, and connections not mentioned in the known facts.

            Known facts about {entity_name}:
            {context}

            Extract 3-5 additional important facts or relationships.
            Be specific and factual.
            """
        )

        try:
            # Extract graph
            graph_documents = self.graph_transformer.convert_to_graph_documents([document])

            new_triplets = []
            for graph_doc in graph_documents:
                for triplet in graph_doc.relationships:
                    new_triplets.append({
                        'head': triplet.source.id,
                        'relation': triplet.type,
                        'tail': triplet.target.id
                    })

            logger.info(f"✅ Extracted {len(new_triplets)} new triplets for {entity_name}")
            return new_triplets

        except Exception as e:
            logger.error(f"Failed to enrich {entity_name}: {e}")
            return []

    def batch_enrich_entities(self, entities_df: pd.DataFrame,
                            triplets_df: pd.DataFrame,
                            sample_size: int = 100) -> pd.DataFrame:
        """
        Enrich a sample of entities with LLM-extracted knowledge

        Args:
            entities_df: DataFrame with entities
            triplets_df: DataFrame with existing triplets
            sample_size: Number of entities to enrich

        Returns:
            DataFrame with new triplets
        """
        # Sample entities (prioritize those with fewer existing triplets)
        entity_counts = triplets_df['head_name'].value_counts()
        sample_entities = entity_counts.nsmallest(sample_size).index.tolist()

        all_new_triplets = []

        for entity in sample_entities:
            # Get existing triplets for this entity
            existing = triplets_df[
                (triplets_df['head_name'] == entity) |
                (triplets_df['tail_name'] == entity)
            ]['triplet_text'].tolist()

            # Enrich
            new_triplets = self.enrich_entity(entity, existing)
            all_new_triplets.extend(new_triplets)

        # Convert to DataFrame
        enriched_df = pd.DataFrame(all_new_triplets)
        logger.info(f"✅ Generated {len(enriched_df)} enriched triplets")
        return enriched_df